import torch
import torch.nn.init as init
from torch.nn import Conv2d
import numpy as np
from builder import ConvBuilder
import torch.nn as nn

class AOFPConvBNLayer(torch.nn.Module):

    def __init__(self, conv_idx, builder:ConvBuilder, preced_layer_idx,
                 score_preced_layer, scored_by_follow_layer, iters_per_half, thresh,
                 in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
        super(AOFPConvBNLayer, self).__init__()
        self.conv_idx = conv_idx
        self.base_path = builder.OriginConv2dBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                                stride=stride, padding=padding, dilation=dilation, groups=groups,
                                                padding_mode=padding_mode)

        if score_preced_layer:
            self.register_buffer('t_value', torch.zeros(1))
        if scored_by_follow_layer:
            self.register_buffer('half_start_iter', torch.zeros(1) + 9999999)
            self.register_buffer('base_mask', torch.ones(out_channels))
            self.register_buffer('score_mask', torch.ones(out_channels))
            self.register_buffer('search_space', torch.ones(out_channels))  # 1 indiates "in the search space"
            # self.reset_search_space()
            self.register_buffer('accumulated_t', torch.zeros(out_channels))
            self.register_buffer('accumulated_cnt', torch.zeros(out_channels))

        self.post = nn.Identity()
        self.preced_layer_idx = preced_layer_idx
        self.scored_by_follow_layer = scored_by_follow_layer
        self.score_preced_layer = score_preced_layer
        self.num_filters = out_channels
        self.iters_per_half = iters_per_half
        self.thresh = thresh
        self.aofp_started = False

    def add_module(self, name: str, module: 'Module') -> None:
        self.post = module

    def forward(self, inputs):
        if self.score_preced_layer:
            base_path_input = inputs['base{}'.format(self.preced_layer_idx)]
            base_path_out = self.post(self.base_path(base_path_input))
            ablate_path_input = inputs['ablate{}'.format(self.preced_layer_idx)]
            ablate_path_out = self.post(self.base_path(ablate_path_input))
            if hasattr(self, 'base_mask'):
                base_path_out = base_path_out * self.base_mask.view(1, -1, 1, 1)
                ablate_path_out = ablate_path_out * self.base_mask.view(1, -1, 1, 1)

            t_value = ((base_path_out.detach() - ablate_path_out.detach()) ** 2).sum() / (base_path_out.detach() ** 2).sum()
            self.t_value = t_value
        else:
            base_path_out = self.post(self.base_path(inputs))
            if hasattr(self, 'base_mask'):
                base_path_out = base_path_out * self.base_mask.view(1, -1, 1, 1)

        if self.scored_by_follow_layer:
            if self.aofp_started:
                self.random_generate_mask()
            return {
                'base{}'.format(self.conv_idx) : base_path_out,
                'ablate{}'.format(self.conv_idx) : base_path_out * self.score_mask.view(1, -1, 1, 1)
            }
        else:
            return base_path_out

    def reset_search_space(self):
        self.search_space = (self.base_mask == 1).type(torch.IntTensor)

    def start_aofp(self, cur_iter):
        print('start aofp on ', self.conv_idx)
        if hasattr(self, 'base_mask'):
            self.aofp_started = True
            self.half_start_iter[0] = cur_iter
            self.reset_search_space()

    def search_space_size(self):
        return self.search_space.sum().cpu().item()

    def random_generate_mask(self):
        search_space_np = self.search_space.cpu().numpy()
        granu = np.sum(search_space_np) // 2
        assert granu > 0
        self.score_mask.copy_(self.base_mask)
        unmasked = np.where(search_space_np == 1)[0]
        picked = np.random.choice(unmasked, granu, replace=False)
        self.score_mask[picked] = 0



    def accumulate_t_value(self, t_value):
        # print('score mask')
        # print(self.score_mask)
        # print(self.search_space == 1)
        # print(self.score_mask == 0)
        t_related = (self.search_space.cpu() == 1) & (self.score_mask.cpu() == 0)
        self.accumulated_t[t_related] += t_value
        self.accumulated_cnt[t_related] += 1

        # for i in range(self.num_filters):
        #     # print('i={}, t={}, search={}, score={}'.format(i, t_value, self.search_space[i], self.score_mask[i]))
        #     if self.search_space[i] == 1 and self.score_mask[i] == 0:
        #         self.accumulated_t[i] += t_value
                # print('accumulate t={} for filter {}'.format(t_value, i))
        # print(self.accumulated_t)

    def finished_a_half(self, iteration):
        if iteration - self.half_start_iter > self.iters_per_half:
            raise ValueError('???')
        return iteration - self.half_start_iter == self.iters_per_half

    def get_averaged_accumulated_t_vector(self):
        t_vector = self.accumulated_t.cpu().numpy()
        cnt_vector = self.accumulated_cnt.cpu().numpy()
        # if self.conv_idx == 0:
        #     print('before: ', cnt_vector)
        t_vector[cnt_vector == 0] = 999
        cnt_vector[cnt_vector == 0] = 0.001
        # if self.conv_idx == 0:
        #     print('after: ', cnt_vector)
        return t_vector / cnt_vector


    def halve_search_space(self, iteration, t_vector, granu):
        sorted_idx = np.argsort(t_vector)
        for i in range(self.num_filters):
            if i in sorted_idx[:granu]:
                self.search_space[i] = 1
            else:
                self.search_space[i] = 0
        self.half_start_iter[0] = iteration

    def mask_cur_granu_and_finish_a_move(self, iteration, t_vector, granu):
        sorted_idx = np.argsort(t_vector)
        print('mask these of conv {} : '.format(self.conv_idx), sorted_idx[:granu])
        for i in range(self.num_filters):
            if i in sorted_idx[:granu]:
                self.base_mask[i] = 0
        base_mask_value = self.base_mask.cpu().numpy()
        zero_idx = np.where(base_mask_value == 0)[0]
        print('conv {} pruned {}, they are {}'.format(self.conv_idx, len(zero_idx), zero_idx))
        self.start_aofp(iteration)

    def halve_or_stop(self, iteration):
        granu = self.search_space_size() // 2
        print('granu of conv {} is {}'.format(self.conv_idx, granu))
        # print(self.search_space)
        assert granu > 0
        t_vector = self.get_averaged_accumulated_t_vector()
        torch.zero_(self.accumulated_t)
        torch.zero_(self.accumulated_cnt)
        sorted_t_vector = sorted(t_vector)
        # print('sorted t vector of layer {} is '.format(self.conv_idx), sorted_t_vector)
        if sorted_t_vector[granu - 1] < self.thresh:
            self.mask_cur_granu_and_finish_a_move(iteration, t_vector, granu)
        elif granu > 1:
            self.halve_search_space(iteration, t_vector, granu)
        else:
            self.start_aofp(iteration)
