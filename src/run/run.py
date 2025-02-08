import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)    #初始化runner对象

    # Set up schemes and groups here
    env_info = runner.get_env_info()    #得到运行的地图的信息，包含‘state_shape’ 'obs_shape' 'n_actions' 'n_agents' 'episode_limit'
    args.n_agents = env_info["n_agents"]    #args增加n_agents变量，表示智能体数量
    args.n_actions = env_info["n_actions"]    #args增加n_actions变量，表示智能体动作数量
    args.state_shape = env_info["state_shape"]    #args增加state_shape变量，表示状态形状
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)  #QMIX没有这个变量

    if getattr(args, 'agent_own_state_size', False):    #QMIX没有这个变量
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme  默认方案    vshape=value shape , dtype=data type    #加task_id
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }    #一个包含OneHot对象的列表。OneHot是一个预处理步骤，用于将整数编码的动作转换为独热编码

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)    #初始化buffer对象  改对象，
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)    #根据算法，设置mac对象，QMIX为n_mac

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)    #给runner对象具体参数进行初始化

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)    #初始化learner对象

    if args.use_cuda:    #QMIX为False
        learner.cuda()
    # 如果有已经保存的模型，就读取此模型，接着训练  ，这一段不用看
    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0
        # 如果路径不存在就会结束run函数，并报错路径不存在
        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return
        # 检测路径文件夹下的文件夹（就是那些以数字命名的文件夹1，2，3，4，5，6......）
        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step#从timesteps列表中找出与args.load_step最接近的timestep
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training 开始训练    从这里继续看
    episode = 0
    last_test_T = -args.test_interval - 1   #初始化last_test_T为负数,防止一开始就很大
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time

        with th.no_grad():    #th.no_grad()是一个上下文管理器，用于禁止在其内部执行的代码块中的所有操作跟踪计算历史。意味着，当在这个上下文中进行的前向传播（forward pass）时，不会计算任何关于输入的梯度。
            episode_batch = runner.run(test_mode=False)    #运行一个训练回合，并返回该回合的数据
            buffer.insert_episode_batch(episode_batch)    #收集到的回合数据插入到一个缓存（buffer）中

        if buffer.can_sample(args.batch_size):    #检查缓存中是否有足够的数据来采样一个批次。如果可以采样，代码接着从缓存中采样一个数据批次，并处理该批次数据，使其适合训练。
            next_episode = episode + args.batch_size_run    #更新回合数
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)    #对buffer采样

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()    #
            episode_sample = episode_sample[:, :max_ep_t]    #截断

            if episode_sample.device != args.device:    #如果采样不在args.device,=，则移到args.device
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)    #用采样到的数据进行训练。
            del episode_sample    #删除采样到的数据，以释放内存。

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)    #计算测试运行次数n_test_runs。= max(1,32/8) = 4
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:    #根据一定的时间间隔args.test_interval，执行测试运行。

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))    #在终端显示信息
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)    #在测试模式下运行一个回合。测试4次

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):    #如果设置了保存模型（args.save_model），并且达到保存间隔或模型尚未保存过，则保存模型。
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)    #将模型保存到指定的路径。

        episode += args.batch_size_run    #更新回合数episode。

        if (runner.t_env - last_log_T) >= args.log_interval:    #如果达到日志记录间隔，记录当前的统计信息并打印最近的统计信息。
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
