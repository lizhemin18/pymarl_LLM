import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
import math

class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(args.rnn_hidden_dim + input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # self.fc_con = nn.Linear(args.n_agents * input_shape, input_shape)

        #self.score = nn.Linear(input_shape, 1)
        #self.sigmoid = nn.Sigmoid()
        self.h2mu = nn.Linear(input_shape, 1)
        self.h2logvar = nn.Linear(input_shape, 1)

        self.fc_neigh = nn.Linear(args.n_agents * input_shape, args.rnn_hidden_dim)
        self.rnn_neigh = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_neigh = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.dropout = nn.Dropout(p=0.5)


        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def init_hidden_2(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, visible_matrix, hidden_state, hidden_state_2,  test_mode=False):
        b, a, e = inputs.size()
        # print("1.", inputs[0])
        #weight_ = self.score(inputs)
        #weight = self.sigmoid(weight_).squeeze(-1)  # 8*5
        
        mu = self.h2mu(inputs)
        logvar = self.h2logvar(inputs)

        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        weight = (mu + std * eps).squeeze(-1).view(b,-1)

        '''		
		# Robust Testing (Test_mode = True)
		mask_percentile = 0.3

        # randomly disconnection
        mask_agent = random.sample(list(range(a)), math.ceil(a * 0.3))
        weight[:,mask_agent] = 0

        # top-k disconnection
		k = np.floor(a*mask_percentile)
        _, indices = torch.topk(weight.view(b, -1), k, dim=1)
		mask = torch.ones_like(weight.view(b, -1))
		mask.scatter_(1, indices, 0)
        weight *= mask.view(b, a, 1)

		'''
        
        weight_copy = weight.unsqueeze(1).repeat(1, a, 1)

        visible_weight = th.mul(weight_copy, visible_matrix)
        
        comm_rate = 0.7
        floor_k_indices = th.topk(visible_weight, k=math.ceil((a-1) * (1-comm_rate)), dim=2, largest=False)[1]

        visible_matrix.scatter_(2, floor_k_indices, 0)
        visible_matrix = visible_matrix + th.eye(a).unsqueeze(0).repeat(b, 1, 1).to(self.args.device)

        expanded_visible_matrix = th.diag_embed(visible_matrix).view(b * a, a, -1)

        neighbor_ = inputs.unsqueeze(1).repeat(1, a, 1, 1).view(b * a, a, -1)

        # print("2.", floor_k_indices[0])


        neighbor = th.bmm(expanded_visible_matrix, neighbor_)
        # print("3.", neighbor[0])
        # print("4.", neighbor.size())
        neighbor = neighbor.view(b * a, -1)

        # neighbor_inputs = self.fc_con(neighbor)
        neighbor_inputs_ = self.fc_neigh(neighbor)
        neighbor_inputs_ = self.dropout(neighbor_inputs_)
        neighbor_inputs = F.relu(neighbor_inputs_)
        h_neigh = hidden_state_2.reshape(-1, self.args.rnn_hidden_dim)
        hh_neigh = self.rnn_neigh(neighbor_inputs, h_neigh)
        # neighbor_inputs_3 = self.fc2_neigh(neighbor_inputs)
        neighbor_inputs_3 = self.fc2_neigh(hh_neigh)
        neighbor_inputs_3 = self.dropout(neighbor_inputs_3)




        inputs = inputs.view(-1, e)
        new_inputs = th.cat((inputs, neighbor_inputs_3), dim=-1)

        x = F.relu(self.fc1(new_inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1), hh_neigh.view(b, a, -1)
