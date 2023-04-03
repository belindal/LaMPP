# code from Junxian He, https://github.com/jxhe/struct-learning-with-flow/blob/master/modules/projection.py
from __future__ import print_function
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLUNet(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--flow_hidden_layers', type=int, default=1)
        parser.add_argument('--flow_hidden_units', type=int, default=100)

    def __init__(self, args, in_features, out_features):
        super(ReLUNet, self).__init__()

        self.args = args

        self.in_layer = nn.Linear(in_features, self.args.flow_hidden_units, bias=True)
        self.out_layer = nn.Linear(self.args.flow_hidden_units, out_features, bias=True)
        for i in range(self.args.flow_hidden_layers):
            name = 'cell{}'.format(i)
            cell = nn.Linear(self.args.flow_hidden_units, self.args.flow_hidden_units, bias=True)
            setattr(self, name, cell)

    def reset_parameters(self):
        self.in_layer.reset_parameters()
        self.out_layer.reset_parameters()
        for i in range(self.args.flow_hidden_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).reset_parameters()

    def init_identity(self):
        self.in_layer.weight.data.zero_()
        self.in_layer.bias.data.zero_()
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()
        for i in range(self.args.flow_hidden_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).weight.data.zero_()
            getattr(self, name).bias.data.zero_()

    def forward(self, input):
        """
        input: (batch_size, seq_length, in_features)
        output: (batch_size, seq_length, out_features)
        """
        h = self.in_layer(input)
        h = F.relu(h)
        for i in range(self.args.flow_hidden_layers):
            name = 'cell{}'.format(i)
            h = getattr(self, name)(h)
            h = F.relu(h)
        return self.out_layer(h)


class NICETrans(nn.Module):
    @classmethod
    def add_args(cls, parser):
        ReLUNet.add_args(parser)
        parser.add_argument('--flow_couple_layers', type=int, default=4)
        parser.add_argument('--flow_scale', action='store_true')
        parser.add_argument('--flow_scale_no_zero', action='store_true')

    def __init__(self,
                 args,
                 features):
        super(NICETrans, self).__init__()

        self.args = args

        for i in range(self.args.flow_couple_layers):
            name = 'cell{}'.format(i)
            cell = ReLUNet(args, features//2, features//2)
            setattr(self, name, cell)
            if args.flow_scale:
                name = 'scale_cell{}'.format(i)
                cell = ReLUNet(args, features//2, features//2)
                if not args.flow_scale_no_zero:
                    cell.init_identity()
                setattr(self, name, cell)

    def reset_parameters(self):
        for i in range(self.args.flow_couple_layers):
            name = 'cell{}'.format(i)
            getattr(self, name).reset_parameters()
            if self.args.flow_scale:
                name = 'scale_cell{}'.format(i)
                getattr(self, name).reset_parameters()


    def forward(self, input):
        """
        input: (batch_size, seq_length, features)
        h: (batch_size, seq_length, features)
        """

        # For NICE it is a constant
        jacobian_loss = torch.zeros(input.size(0), device=input.device, requires_grad=False)

        ep_size = input.size()
        features = ep_size[-1]
        # h = odd_input
        h = input
        for i in range(self.args.flow_couple_layers):
            name = 'cell{}'.format(i)
            h1, h2 = torch.split(h, features//2, dim=-1)
            if i%2 == 1:
                h1, h2 = h2, h1
            t = getattr(self, name)(h1)
            if self.args.flow_scale:
                s = getattr(self, 'scale_cell{}'.format(i))(h1)
                jacobian_loss += s.sum(dim=-1).sum(dim=-1)
                h2_p = torch.exp(s) * h2 + t
            else:
                h2_p = h2 + t
            if i%2 == 1:
                h1, h2_p = h2_p, h1
            h = torch.cat((h1, h2_p), dim=-1)
            # if i%2 == 0:
            #     h = torch.cat((h1, h2 + getattr(self, name)(h1)), dim=-1)
            # else:
            #     h = torch.cat((h1 + getattr(self, name)(h2), h2), dim=-1)
        return h, jacobian_loss
