import torch
import torch.nn.init as init
from torch.nn import Conv2d
import numpy as np
from builder import ConvBuilder
import torch.nn as nn

class AOFPFCReluLayer(torch.nn.Module):

    def __init__(self, conv_idx, builder:ConvBuilder, preced_layer_idx,
                 in_features, out_features, bias=True):
        super(AOFPFCReluLayer, self).__init__()
        self.conv_idx = conv_idx
        self.base_path = builder.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.relu = builder.ReLU()
        self.register_buffer('t_value', torch.zeros(1))
        self.preced_layer_idx = preced_layer_idx

    def forward(self, inputs):
        base_path_input = inputs['base{}'.format(self.preced_layer_idx)]
        base_path_out = self.relu(self.base_path(base_path_input))
        ablate_path_input = inputs['ablate{}'.format(self.preced_layer_idx)]
        ablate_path_out = self.relu(self.base_path(ablate_path_input))
        t_value = ((base_path_out.detach() - ablate_path_out.detach()) ** 2).sum() / (base_path_out.detach() ** 2).sum()
        self.t_value = t_value
        return base_path_out