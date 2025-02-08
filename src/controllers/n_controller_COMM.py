from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np
import time

# This multi-agent controller shares parameters between agents
class NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMAC, self).__init__(scheme, groups, args)
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # print(t_ep)
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # print(qvals.shape)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()
            
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        visible_matrix = ep_batch["visible_matrix"][:, t]
        # print(adjacent_t.shape)
        # start = time.time()
        # print(t)
        agent_outs, self.hidden_states, self.hidden_states_2 = self.agent(agent_inputs, visible_matrix, self.hidden_states, self.hidden_states_2, test_mode = test_mode)
        # end = time.time()
        # spend = end - start
        # print(spend)
        
        return agent_outs
