import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer

from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num

class NQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':    #QMIX为adam
            self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)    #QMIX为False
        self.return_priority = getattr(self.args, "return_priority", False)    #QMIX为False
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]  #一个torch 第一维度的全部，第二维度除了最后一个   形状为[b,t-1,1]
        actions = batch["actions"][:, :-1]    #形状为[b,t-1,n_agents,1]
        terminated = batch["terminated"][:, :-1].float()    #将数据类型转换为了float类型    形状为[b,t-1,1]
        mask = batch["filled"][:, :-1].float()    #形状为[b,t-1,1]
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])    #只保留未终止时的数据，若终止则mask对应元素为0
        avail_actions = batch["avail_actions"]    #形状为[b,t,n_agents,n_actions]

        # Calculate estimated Q-Values
        self.mac.agent.train()    # 将模型设置为训练模式,影响某些层的行为
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)    #agent_out = torch.size([b,a,-1]),记录每个智能体采取各个动作会得到的Q值    这里是否仅仅是把抽到的episode在各个时刻的动作和状态都输入agent网络来得到采样过程的q值？？？
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time    #假设mac_out是一个张量列表，每个张量的形状是[batch_size, features]。那么执行th.stack(mac_out, dim=1)之后，得到的mac_out将是一个形状为[batch_size, len(mac_out), features]的三维张量。
        #从[b,a,-1]变为[b,t,a,-1], mac_out存储的是每个batch每个时刻每个智能体可以采取的所有动作的Q值向量，chosen_action_qvals存储的是在采样过程中执行的那个动作所得到的Q值
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim    chosen_action_qvals将包含根据actions索引从mac_out[:, :-1]中收集到的Q值。如果mac_out是一个包含多个智能体、多个时间步和多个可能动作的Q值张量，那么chosen_action_qvals将包含每个智能体在每个时间步选择特定动作对应的Q值。
        #这行代码通常出现在强化学习的上下文中，特别是在使用多智能体合作（Multi - AgentCooperation, MAC）方法时。在这种情况下，mac_out可能包含了多个智能体在每个时间步对于每个可能动作的Q值估计，而actions则包含了每个智能体在每个时间步实际选择的动作索引。通过gather和squeeze操作，代码提取了实际选择动作对应的Q值。
        #从[b,t,a,-1]变为[b,t,a]
        chosen_action_qvals_ = chosen_action_qvals

        # Calculate the Q-Values necessary for the target
        with th.no_grad():    #一个上下文管理器，它用于指示在该上下文块内的所有操作不应该计算梯度
            self.target_mac.agent.train()    #设置目标网络神经网络为训练模式
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)    #保存每个时刻t的所有智能体采取各个动作会得到的Q值

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time    target_mac_out = torch.Size([128, 71, 5, 12]) 71是该batch的episode里的最长时间步    【b,a,-1】变为[b,t,a,-1]
            ##假设mac_out是一个张量列表，每个张量的形状是[batch_size, features]。那么执行th.stack(mac_out, dim=1)之后，得到的mac_out将是一个形状为[batch_size, len(mac_out), features]的三维张量。
            # Max over target Q-Values/ Double q learning    最大化目标Q值
            mac_out_detach = mac_out.clone().detach()    #创建一个与 mac_out 具有相同数据的新张量，但这个新张量不会保留计算历史或梯度。(独立于其计算历史)
            mac_out_detach[avail_actions == 0] = -9999999    #将不可行的动作的Q值设为-9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]    #一个与 mac_out_detach 在前三个维度上大小相同的张量，但在第四个维度上大小为1（由于 keepdim=True）。这个张量包含了每个智能体在每个时间步可以选择的Q值最大的动作的索引。
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)    #通过gather和squeeze来保存每个智能体采取最优动作得到的Q值   存储的是对于采样的数据如果每个智能体在每个时刻采取Q值最大的最优动作该智能体会得到的Q值 形状为[b,t,a]

            # Calculate n-step Q-Learning targets    计算n步Q-Learning 目标
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])    #将每个时刻可得的最大Q值和每个时刻的全局状态输入混合网络    从[128,71,5]变为[128,71,1]


            if getattr(self.args, 'q_lambda', False):    #False
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)    #对td_error的每个元素二次方再乘0.5

        mask = mask.expand_as(td_error2)    #扩展mask的形状以匹配td_error2的形状
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:    #QMIX为False
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        loss = L_td = masked_td_error.sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()    #清除梯度
        loss.backward()    #反向传播：计算梯度
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()    #使用优化器更新权重

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
