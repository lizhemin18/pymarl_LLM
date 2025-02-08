from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args    #增加属性args
        self.logger = logger    #增加实例属性logger
        self.batch_size = self.args.batch_size_run    #增加属性 batch_size,8

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])    #创建了self.batch_size个独立的管道对象对,利用zip，*操作来将所有管道的发送端存储在self.parent_conns中，将所有管道的接收端存储在self.worker_conns中
        env_fn = env_REGISTRY[self.args.env]    #env_fn是一个部分应用的env_fn函数，因为env固定，**kwargs不定
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):    #为self.worker_conns中的每个连接对象创建一个子进程，这些子进程将运行env_worker函数，并接收一个连接对象和一个部分应用的env_fn函数作为参数。
            ps = Process(target=env_worker, 
                    args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)

        for p in self.ps:    #遍历所有创建的子进程，将它们设置为守护进程，并启动它们。
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()    #初始化runner的batch属性

        # Reset the envs    重置环境
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))    #通过每个parent_conn连接对象发送一个消息。消息是一个元组

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back   接受环境最初的全局状态， avail_actions, obs信息
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()    #parent_conn.recv() = dict_keys(['state', 'avail_actions', 'obs'])
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)    #更新batch

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]    #存储batch每个episode的returns
        episode_lengths = [0 for _ in range(self.batch_size)]    #存储batch每个episode的length
        self.mac.init_hidden(batch_size=self.batch_size)    #为mac对象增加属性hidden_states
        terminated = [False for _ in range(self.batch_size)]    #存储batch每个episode是否终止
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]    #存储不暂停的batch的id
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        
        save_probs = getattr(self.args, "save_probs", False)    #QMIX中是false
        while True:

            # Pass the entire batch of experiences up till now to the agents 将迄今为止的全部经历传递给智能体
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env  在每个未终止的env的批处理中接收每个智能体在此时间步长的操作
            if save_probs: #不用看
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            else:    #看这个
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)    #对该批次智能体选择动作

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken 更新所采取的操作
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }    #将actions从[n,5]转换为[n,1,5]

            if save_probs:    #不用看
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")
            # bs是还没停止的episode序号，ts是该batch的运行时间
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0    #初始化一个动作索引，用于跟踪当前要发送的动作在cpu_actions中的位置
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)    #使用all函数检查terminated列表中的所有元素是否都为True
            if all_terminated:    #如果all_terminated为True，即所有环境都已终止，则执行break语句。这将跳出当前的循环或循环结构，不再继续执行后续的代码。
                break

            # Post step data we will insert for the current timestep 当前时间步要插入的时间步数据
            post_transition_data = {
                "reward": [],
                "terminated": []
            }    #转换后数据
            # Data for the next step we will insert in order to select an action  下一时间步我们将要插入用来获得一个动作的数据
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }    #转换前数据

            # Receive data back for each unterminated env    从每个未终止的环境接受数据
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()    #data = dict_keys(['state', 'avail_actions', 'obs', 'reward', 'terminated', 'info'])
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]    #将这一步的reward加到这个episdoe的return
                    episode_lengths[idx] += 1    #这个episode的length增加1
                    if not test_mode:    #如果不是test，则本次运行的环境的时间步增加1
                        self.env_steps_this_run += 1

                    env_terminated = False    #环境未终止
                    if data["terminated"]:    #如果这一步环境终止，则添加info到final_env_infos
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):    #如果环境终止，但是没有设置episode_limit
                        env_terminated = True
                    terminated[idx] = data["terminated"]    #记录这个episode此时是否终止
                    post_transition_data["terminated"].append((env_terminated,))    #将元组(env_terminated,)添加到post_transition_data字典中键为"terminated"的列表的末尾。

                    # Data for the next timestep needed to select an action 下一步用來选择动作的数据
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])


            # Add post_transiton data into the batch  添加转换前数据到batch里
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []    #不懂这有啥用？
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

