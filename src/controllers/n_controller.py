from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMAC, self).__init__(scheme, groups, args)
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]    #此处别有玄机，batch对象通过[]来访问，实则是调用了内部的__getitem__(self.item)函数
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)    #对于epsilongGreedy算法，若test_mode为True,则epsilong = 0
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]    #没用？
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs