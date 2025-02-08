import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):    #False
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):    #False
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()    #b是batch_size, a是n_agents, e是obs和actions_onehot的拼凑维度

        inputs = inputs.view(-1, e)    #这样，第一维是整个batch的所有智能体数量，第二维对应每个智能体的输入
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)    #hidden_state从[batch_size, self.n_agents, self.args.rnn_hidden_dim]重塑为[-1,self.args.rnn_hidden_dim]
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):    #QMIX为False
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)
