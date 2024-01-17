import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Truncated_power(nn.Module):
    def __init__(self, degree, knots):
        super(Truncated_power, self).__init__()
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis, device=x.device)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                out[:, _] = x ** _
            else:
                out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out


class MLP_treatnet(nn.Module):
    def __init__(self, num_out, n_hidden=10, num_in=4) -> None:
        super(MLP_treatnet, self).__init__()
        self.num_in = num_in
        self.hidden1 = torch.nn.Linear(num_in, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, num_out)
        self.act = nn.ReLU()

    def forward(self, x):
        x_mix = torch.zeros([x.shape[0], 3])
        x_mix = x_mix.to(x.device)
        x_mix[:, 0] = 0

        x_mix[:, 1] = torch.cos(x * np.pi)
        x_mix[:, 2] = torch.sin(x * np.pi)
        h = self.act(self.hidden1(x_mix))  # relu
        y = self.predict(h)

        return y


class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0, dynamic_type='power'):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        if dynamic_type == 'power':
            self.spb = Truncated_power(degree, knots)
            self.d = self.spb.num_of_basis  # num of basis
        else:
            self.spb = MLP_treatnet(num_out=10, num_in=3)
            self.d = 10

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'soft':
            self.act = nn.Softplus()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]

        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T  # bs, outd, d

        x_treat_basis = self.spb(x_treat)  # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2)  # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out


# ------------propensity------------

def init_weights(m):
    if isinstance(m, nn.Linear):
        stdv = 1 / math.sqrt(m.weight.size(1))
        torch.nn.init.normal_(m.weight, mean=0.0, std=stdv)
        # torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class BaseModel(nn.Module):
    def __init__(self, input_dim, base_dim, cfg):
        super(BaseModel, self).__init__()
        self.SpNN = nn.Sequential(
            nn.Linear(input_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=0.01),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=0.01),
            nn.Linear(base_dim, base_dim),
            # nn.BatchNorm1d(base_dim),
            nn.ELU(),
            nn.Dropout(p=0.01)
        )
        self.SpNN.apply(init_weights)

    def forward(self, x):  # input->[500,1]
        logits = self.SpNN(x)
        return logits


class PropstyNetwork(nn.Module):
    """propensity network"""

    def __init__(self, input_dim, base_dim, cfg):
        super(PropstyNetwork, self).__init__()
        self.baseModel = BaseModel(input_dim, base_dim, cfg)
        self.logitLayer = nn.Linear(base_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.logitLayer.apply(init_weights)

    def forward(self, inputs):
        inputs = self.baseModel(inputs)
        p = self.logitLayer(inputs)
        return p


# Targeted Regularizer
class TR(nn.Module):
    def __init__(self, degree, knots):
        super(TR, self).__init__()
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis  # num of basis
        self.weight = nn.Parameter(torch.rand(self.d), requires_grad=True)

    def forward(self, t):
        out = self.spb.forward(t)
        out = torch.matmul(out, self.weight)
        return out

    def _initialize_weights(self):
        # self.weight.data.normal_(0, 0.01)
        self.weight.data.zero_()
