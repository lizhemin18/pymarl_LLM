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
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        # if t_env < 280 and test_mode == False:
        #     # print(t_env)
        #     qvals = self.llmforward(ep_batch, t_ep, test_mode=test_mode)
        #     chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=True)
        #
        # else:

        # qvals_, q_LLM = self.forward(ep_batch, t_ep, epsilon = False, llm=True, test_mode=test_mode)


        epsilon = self.action_selector.select_action(avail_actions[bs], avail_actions[bs], avail_actions[bs], t_env, out_ep=True, test_mode=test_mode)

        qvals, q_LLM = self.forward(ep_batch, t_ep, epsilon, llm=True, test_mode=test_mode)

        chosen_actions = self.action_selector.select_action(qvals[bs], q_LLM[bs], avail_actions[bs], t_env, out_ep = False, test_mode=test_mode)

        return chosen_actions

    def forward(self, ep_batch, t, epsilon, llm, test_mode=False):
        if test_mode:
            self.agent.eval()

        coord = ep_batch["coord"][:, t]
        state = ep_batch["state"][:, t]

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states, q_LLM = self.agent(agent_inputs, state, self.hidden_states, coord, llm, epsilon, test_mode=False)

        return agent_outs, q_LLM

    # def llmforward(self, ep_batch, t, test_mode=False):
    #     if test_mode:
    #         self.agent.eval()
    #
    #     coord = ep_batch["coord"][:, t]
    #
    #     agent_inputs = self._build_inputs(ep_batch, t)
    #     agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, coord, llm=True, test_mode=False)
    #
    #     return agent_outs
