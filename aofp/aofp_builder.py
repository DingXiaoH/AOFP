from rr.resrep_config import ResRepConfig
from builder import ConvBuilder
from base_config import BaseConfigByEpoch
import torch.nn as nn
from aofp.aofp_conv import AOFPConvBNLayer
from aofp.other_layers import *
from aofp.aofp_fcrelu import AOFPFCReluLayer

class AOFPBuilder(ConvBuilder):

    def __init__(self, base_config:BaseConfigByEpoch,
                 target_layers, succ_strategy,
                 iters_per_half, thresh):
        super(AOFPBuilder, self).__init__(base_config=base_config)
        self.target_layers = target_layers
        self.succ_strategy = succ_strategy
        self.iters_per_half = iters_per_half
        self.thresh = thresh
        self.preced_strategy = {v:k for k, v in succ_strategy.items()}    # the k-th layer is followed by the v-th layer only
        print('succ strategy', self.succ_strategy)
        print(self.preced_strategy)


    def Conv2dBN(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
               padding_mode='zeros', use_original_conv=False):
        self.cur_conv_idx += 1
        assert type(kernel_size) is int
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        preced_layer_idx = self.preced_strategy[self.cur_conv_idx] if self.cur_conv_idx in self.preced_strategy else None
        score_preced_layer = preced_layer_idx is not None and preced_layer_idx in self.target_layers

        scored_by_follow_layer = self.cur_conv_idx in self.target_layers
        if scored_by_follow_layer:
            print('conv {} is scored by follow layer'.format(self.cur_conv_idx))
        else:
            print('conv {} is not scored by any layer'.format(self.cur_conv_idx))
        if score_preced_layer:
            print('conv {} scores conv {}'.format(self.cur_conv_idx, self.preced_strategy[self.cur_conv_idx]))

        return AOFPConvBNLayer(conv_idx=self.cur_conv_idx, builder=self,
                               preced_layer_idx=preced_layer_idx,
                               score_preced_layer=score_preced_layer,
                               scored_by_follow_layer=scored_by_follow_layer,
                               iters_per_half=self.iters_per_half, thresh=self.thresh,
                               in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)

    def Maxpool2d(self, kernel_size, stride=None, padding=0):
        return AOFPMaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def Flatten(self):
        return AOFPFlatten()

    def IntermediateLinear(self, in_features, out_features, bias=True):

        if self.base_config.network_type == 'vc' and 12 in self.target_layers:
            print('---------------')
            print('------ aofp for conv 12 of vc')
            print('----------------------------------------------')
            return AOFPFCReluLayer(conv_idx=13, builder=self, preced_layer_idx=12, in_features=in_features,
                                   out_features=out_features, bias=bias)
        elif self.base_config.network_type == 'lenet5bn' and 1 in self.target_layers:
            print('---------------')
            print('------ aofp for conv 1 of lenet')
            print('----------------------------------------------')
            return AOFPFCReluLayer(conv_idx=2, builder=self, preced_layer_idx=1, in_features=in_features,
                                   out_features=out_features, bias=bias)

        else:
            return super(AOFPBuilder, self).IntermediateLinear(in_features=in_features, out_features=out_features, bias=bias)
