# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CNN builder."""

from __future__ import print_function

from collections import defaultdict
import contextlib

import numpy as np

import tensorflow as tf

from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.training import moving_averages
from tensorflow.python.layers.core import dense

class ConvNetBuilder(object):
    """Builder of cnn net."""

    def __init__(self,
                 input_op,
                 input_nchan,
                 phase_train,
                 use_tf_layers,
                 data_format='NCHW',
                 dtype=tf.float32,
                 variable_dtype=tf.float32,
                 use_dense_layer=False,
                 input_rotation=0):
        self.top_layer = input_op
        self.top_size = input_nchan
        self.phase_train = phase_train
        self.use_tf_layers = use_tf_layers
        self.data_format = data_format
        self.dtype = dtype
        self.variable_dtype = variable_dtype
        self.counts = defaultdict(lambda: 0)
        self.use_batch_norm = False
        self.batch_norm_config = {'decay':0.9, 'epsilon':1e-3, 'scale':True}  # 'decay': 0.997, 'scale': True}
        self.channel_pos = ('channels_last'
                            if data_format == 'NHWC' else 'channels_first')
        self.aux_top_layer = None
        self.aux_top_size = 0
        self.need_record_internal_outputs = False
        self.internal_outputs_dict = {}     # {[layer_idx]$[layer_name] : output tensor} for conv, {[layer_name] : output tensor} elsewise
        self.num_internal_conv_outputs = 0

        self.use_dense_layer = use_dense_layer

        if input_rotation > 0:
            print('the input rotation type is ', input_rotation)
        if input_rotation == 1:
            self.top_layer = tf.image.rot90(self.top_layer, k=1)
        elif input_rotation == 2:
            self.top_layer = tf.image.rot90(self.top_layer, k=2)
        elif input_rotation == 3:
            self.top_layer = tf.image.rot90(self.top_layer, k=3)
        elif input_rotation == 4:
            self.top_layer = tf.image.flip_up_down(self.top_layer)
        elif input_rotation == 5:
            self.top_layer = tf.image.flip_left_right(self.top_layer)

    def set_top(self, top_layer, top_size=None):
        self.top_layer = top_layer
        if top_size is not None:
            self.top_size = top_size
        else:
            top_shape = top_layer.get_shape().as_list()
            assert len(top_shape) == 4
            if self.data_format == 'NHWC':
                self.top_size = top_shape[3]
            else:
                self.top_size = top_shape[1]

    def set_default_batch_norm_config(self, decay, epsilon, scale):
        new_config = {'decay':decay, 'epsilon':epsilon, 'scale':scale}
        self.batch_norm_config = new_config
        print('---****---**** reset default BN config', new_config)

    def set_whether_use_batch_norm_by_default(self, use_batch_norm):
        self.use_batch_norm = use_batch_norm

    def pad2d(self, pixel, input_layer=None):
        input = input_layer or self.top_layer
        if self.data_format == 'NHWC':
            output = tf.pad(input, [[0, 0], [pixel,pixel], [pixel, pixel], [0, 0]])
        else:
            output = tf.pad(input, [[0, 0], [0, 0], [pixel, pixel], [pixel, pixel]])
        self.top_layer = output
        return self.top_layer

    def relu(self, input_layer=None):
        if input_layer is None:
            input_layer = self.top_layer
        output = tf.nn.relu(input_layer)
        self.top_layer = output
        return output

    def add(self, x, y):
        return x + y

    def spatial_mean(self, keep_dims=False):
        name = 'spatial_mean' + str(self.counts['spatial_mean'])
        self.counts['spatial_mean'] += 1
        axes = [1, 2] if self.data_format == 'NHWC' else [2, 3]
        self.top_layer = tf.reduce_mean(
            self.top_layer, axes, keepdims=keep_dims, name=name)
        return self.top_layer

    def channel_concat(self, list):
        axis = 3 if self.data_format == 'NHWC' else 1
        return tf.concat(list, axis=axis)


    def scconv_Qd(self, scc_type, bn_type, use_relu, input_layer=None):
        if input_layer is None:
            input_layer = self.top_layer
        name = 'scconv' + str(self.counts['scconv'])
        self.counts['scconv'] += 1
        strides = [1, 1]
        channel_pos = 'channels_first'

        print('input of scconv: ', input_layer.name, input_layer.get_shape())

        with tf.variable_scope(name + '_h'):
            x = tf.pad(input_layer, [[0, 0], [0, 0], [1, 1], [1, 1]])
            x = conv_layers.conv2d(x, filters=input_layer.get_shape()[1].value / 2, kernel_size=3, strides=strides,
                padding='VALID', data_format=channel_pos,
                kernel_initializer=None,
                use_bias=(bn_type == 'NONE'))
            if bn_type != 'NONE':
                x = tf.contrib.layers.batch_norm(
                    x,
                    decay=self.batch_norm_config.get('decay', 0.9),
                    scale=True,
                    epsilon=self.batch_norm_config.get('epsilon', 0.001),
                    is_training=self.phase_train,
                    fused=True,
                    data_format=bn_type)
            if use_relu:
                x = tf.nn.relu(x)

        print('height compressed: ', x.name, x.get_shape())

        with tf.variable_scope(name + '_w'):
            x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]])
            x = conv_layers.conv2d(x, filters=input_layer.get_shape()[2].value / 2, kernel_size=3, strides=strides,
                padding='VALID', data_format=channel_pos,
                kernel_initializer=None,
                use_bias=(bn_type == 'NONE'))
            if bn_type != 'NONE':
                x = tf.contrib.layers.batch_norm(
                    x,
                    decay=self.batch_norm_config.get('decay', 0.9),
                    scale=True,
                    epsilon=self.batch_norm_config.get('epsilon', 0.001),
                    is_training=self.phase_train,
                    fused=True,
                    data_format=bn_type)
            if use_relu:
                x = tf.nn.relu(x)
            x = tf.transpose(x, [0, 2, 1, 3])

        print('width compressed: ', x.name, x.get_shape())
        self.top_layer = x
        return x


    def scconv(self, scc_type, bn_type, use_relu, input_layer=None):
        assert self.data_format == 'NHWC'
        if scc_type == 'Qd':
            return self.scconv_Qd(scc_type=scc_type, bn_type=bn_type, use_relu=use_relu, input_layer=input_layer)
        else:
            assert False











    def enable_record_internal_outputs(self):
        self.need_record_internal_outputs = True
        print('convnet builder: record_internal_outputs enabled')

    def get_internal_outputs_dict(self):
        assert self.need_record_internal_outputs
        print('got {} internal conv outputs and {} outputs in total'.format(self.num_internal_conv_outputs, len(self.internal_outputs_dict)))
        return self.internal_outputs_dict

    def get_custom_getter(self):
        """Returns a custom getter that this class's methods must be called under.

        All methods of this class must be called under a variable scope that was
        passed this custom getter. Example:

        ```python
        network = ConvNetBuilder(...)
        with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
          network.conv(...)
          # Call more methods of network here
        ```

        Currently, this custom getter only does anything if self.use_tf_layers is
        True. In that case, it causes variables to be stored as dtype
        self.variable_type, then casted to the requested dtype, instead of directly
        storing the variable as the requested dtype.
        """

        def inner_custom_getter(getter, *args, **kwargs):
            """Custom getter that forces variables to have type self.variable_type."""
            if not self.use_tf_layers:
                return getter(*args, **kwargs)
            requested_dtype = kwargs['dtype']
            if not (requested_dtype == tf.float32 and
                            self.variable_dtype == tf.float16):
                # Only change the variable dtype if doing so does not decrease variable
                # precision.
                kwargs['dtype'] = self.variable_dtype
            var = getter(*args, **kwargs)
            # This if statement is needed to guard the cast, because batch norm
            # assigns directly to the return value of this custom getter. The cast
            # makes the return value not a variable so it cannot be assigned. Batch
            # norm variables are always in fp32 so this if statement is never
            # triggered for them.
            if var.dtype.base_dtype != requested_dtype:
                var = tf.cast(var, requested_dtype)
            return var

        return inner_custom_getter

    @contextlib.contextmanager
    def switch_to_aux_top_layer(self):
        """Context that construct cnn in the auxiliary arm."""
        if self.aux_top_layer is None:
            raise RuntimeError('Empty auxiliary top layer in the network.')
        saved_top_layer = self.top_layer
        saved_top_size = self.top_size
        self.top_layer = self.aux_top_layer
        self.top_size = self.aux_top_size
        yield
        self.aux_top_layer = self.top_layer
        self.aux_top_size = self.top_size
        self.top_layer = saved_top_layer
        self.top_size = saved_top_size

    def get_variable(self, name, shape, dtype, cast_dtype, *args, **kwargs):
        # TODO(reedwm): Currently variables and gradients are transferred to other
        # devices and machines as type `dtype`, not `cast_dtype`. In particular,
        # this means in fp16 mode, variables are transferred as fp32 values, not
        # fp16 values, which uses extra bandwidth.
        var = tf.get_variable(name, shape, dtype, *args, **kwargs)
        return tf.cast(var, cast_dtype)

    def _crop(self, input, position, pixel):
        if self.data_format == 'NHWC':
            if position == 'up':
                return input[:, -pixel:, :, :]
            elif position == 'bottom':
                return input[:, :pixel, :, :]
            elif position == 'left':
                return input[:, :, -pixel:, :]
            else:
                return input[:, :, :pixel, :]
        else:
            if position == 'up':
                return input[:, :, -pixel:, :]
            elif position == 'bottom':
                return input[:, :, pixel, :]
            elif position == 'left':
                return input[:, :, :, -pixel:]
            else:
                return input[:, :, :, :pixel]

    def _pad(self, input, position, pixel):
        if position == 'up':
            paddings = [[0, 0], [pixel, 0], [0, 0], [0, 0]]
        elif position == 'bottom':
            paddings = [[0, 0], [0, pixel], [0, 0], [0, 0]]
        elif position == 'left':
            paddings = [[0, 0], [0, 0], [pixel, 0], [0, 0]]
        else:
            paddings = [[0, 0], [0, 0], [0, pixel], [0, 0]]
        if self.data_format == 'NCHW':
            paddings.pop(3)
            paddings.insert(1, [0, 0])
        return tf.pad(input, paddings=paddings)

    def _crop_or_pad(self, input, pixel, position):
        assert position in ['up', 'bottom', 'left', 'right']
        if pixel > 0:
            return self._pad(input, position, pixel)
        elif pixel < 0:
            return self._crop(input, position, pixel)
        else:
            return input

    def _conv2d_impl(self, input_layer, num_channels_in, filters, kernel_size,
                     strides, padding, kernel_initializer, specify_padding=None):

        if specify_padding is not None:
            assert padding == 'VALID'
            if hasattr(specify_padding, '__len__'):
                assert len(specify_padding) == 4
                pad_set = specify_padding
            else:
                pad_set = (specify_padding, specify_padding, specify_padding, specify_padding)

            input_layer = self._crop_or_pad(input_layer, pad_set[0], 'up')
            input_layer = self._crop_or_pad(input_layer, pad_set[1], 'bottom')
            input_layer = self._crop_or_pad(input_layer, pad_set[2], 'left')
            input_layer = self._crop_or_pad(input_layer, pad_set[3], 'right')
        #
        # if pad_set[0] > 0:
        #     assert padding == 'VALID'
        #     assert pad_set[1] >= 0
        #     print('PAD HEIGHT: pad the conv input with specified padding=', specify_padding)
        #     if self.channel_pos == 'channels_last':
        #         input_layer = tf.pad(input_layer, [[0, 0], [pad_set[0], pad_set[1]], [0, 0], [0, 0]])
        #     else:
        #         assert self.channel_pos == 'channels_first'
        #         input_layer = tf.pad(input_layer, [[0, 0], [0, 0], [pad_set[0], pad_set[1]], [0, 0]])
        # elif pad_set[0] < 0:
        #     assert padding == 'VALID'
        #     assert pad_set[1] <=0
        #     print('CROP HEIGHT: crop the conv input with specified padding=', specify_padding)
        #     if self.channel_pos == 'channels_last':
        #         input_layer = input_layer[:, -pad_set[0]:pad_set[1], :, :]
        #     else:
        #         assert self.channel_pos == 'channels_first'
        #         input_layer = input_layer[:, :, -pad_set[0]:pad_set[1], :]
        #
        # if pad_set[1] > 0:
        #     assert padding == 'VALID'
        #     print('PAD WIDTH: pad the conv input with specified padding=', specify_padding)
        #     if self.channel_pos == 'channels_last':
        #         input_layer = tf.pad(input_layer, [[0, 0], [0, 0], [pad_set[1], pad_set[1]], [0, 0]])
        #     else:
        #         assert self.channel_pos == 'channels_first'
        #         input_layer = tf.pad(input_layer, [[0, 0], [0, 0], [0, 0], [pad_set[1], pad_set[1]]])
        # elif pad_set[1] < 0:
        #     assert padding == 'VALID'
        #     print('CROP HEIGHT: crop the conv input with specified padding=', specify_padding)
        #     if self.channel_pos == 'channels_last':
        #         input_layer = input_layer[:, :, -pad_set[1]:pad_set[1], :]
        #     else:
        #         assert self.channel_pos == 'channels_first'
        #         input_layer = input_layer[:, :, :, -pad_set[1]:pad_set[1]]
        #
        #
        # if pad_set[0] > 0 or pad_set[1] > 0:
        #     assert padding == 'VALID'
        #     assert pad_set[0] * pad_set[1] >= 0
        #     print('pad the conv input with specified padding=', specify_padding)
        #     if self.channel_pos == 'channels_last':
        #         input_layer = tf.pad(input_layer, [[0, 0], [pad_set[0], pad_set[0]], [pad_set[1], pad_set[1]], [0, 0]])
        #     else:
        #         assert self.channel_pos == 'channels_first'
        #         input_layer = tf.pad(input_layer,
        #             [[0, 0], [0, 0], [pad_set[0], pad_set[0]], [pad_set[1], pad_set[1]]])
        #
        # elif pad_set[0] < 0 or pad_set[1] < 0:
        #     assert padding == 'VALID'
        #     assert pad_set[0] * pad_set[1] >= 0
        #     print('crop the conv input with specified padding=', specify_padding)
        #     if self.channel_pos == 'channels_last':
        #         input_layer = input_layer[:, -pad_set[0]:pad_set[0], -pad_set[1]:pad_set[1], :]
        #     else:
        #         assert self.channel_pos == 'channels_first'
        #         input_layer = input_layer[:, :, -pad_set[0]:pad_set[0], -pad_set[1]:pad_set[1]]
        #
        # else:
        #     assert pad_set[0] == 0, pad_set[1] == 0

        if self.use_tf_layers:
            return conv_layers.conv2d(input_layer, filters, kernel_size, strides,
                padding, self.channel_pos,
                kernel_initializer=kernel_initializer,
                use_bias=False)
        else:
            weights_shape = [kernel_size[0], kernel_size[1], num_channels_in, filters]
            # We use the name 'conv2d/kernel' so the variable has the same name as its
            # tf.layers equivalent. This way, if a checkpoint is written when
            # self.use_tf_layers == True, it can be loaded when
            # self.use_tf_layers == False, and vice versa.
            weights = self.get_variable('conv2d/kernel', weights_shape,
                self.variable_dtype, self.dtype,
                initializer=kernel_initializer)
            if self.data_format == 'NHWC':
                strides = [1] + strides + [1]
            else:
                strides = [1, 1] + strides
            return tf.nn.conv2d(input_layer, weights, strides, padding,
                data_format=self.data_format)

    def conv(self,
             num_out_channels,
             k_height,
             k_width,
             d_height=1,
             d_width=1,
             mode='SAME',
             input_layer=None,
             num_channels_in=None,
             use_batch_norm=None,
             stddev=None,
             activation='relu',
             bias=0.0,
             kernel_initializer=None,
             specify_padding=None,
             name=None,
             name_postfix=None,
             count_convs=True,
             just_as_classic_conv=True):    # just_as_classic_conv is no used by convnet_builder.py
        """Construct a conv2d layer on top of cnn."""
        if input_layer is None:
            input_layer = self.top_layer
        if num_channels_in is None:
            num_channels_in = self.top_size
        if stddev is not None and kernel_initializer is None:
            kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
        if name is None:
            name = 'conv' + str(self.counts['conv'])
        if name_postfix is not None:
            name += name_postfix
        if count_convs:
            self.counts['conv'] += 1
        with tf.variable_scope(name):
            strides = [1, d_height, d_width, 1]
            if self.data_format == 'NCHW':
                strides = [strides[0], strides[3], strides[1], strides[2]]

            if mode != 'SAME_RESNET':
                conv = self._conv2d_impl(input_layer, num_channels_in, num_out_channels,
                    kernel_size=[k_height, k_width],
                    strides=[d_height, d_width], padding=mode,
                    kernel_initializer=kernel_initializer, specify_padding=specify_padding)
            else:  # Special padding mode for ResNet models
                if d_height == 1 and d_width == 1:
                    conv = self._conv2d_impl(input_layer, num_channels_in,
                        num_out_channels,
                        kernel_size=[k_height, k_width],
                        strides=[d_height, d_width], padding='SAME',
                        kernel_initializer=kernel_initializer)
                else:
                    rate = 1  # Unused (for 'a trous' convolutions)
                    kernel_height_effective = k_height + (k_height - 1) * (rate - 1)
                    pad_h_beg = (kernel_height_effective - 1) // 2
                    pad_h_end = kernel_height_effective - 1 - pad_h_beg
                    kernel_width_effective = k_width + (k_width - 1) * (rate - 1)
                    pad_w_beg = (kernel_width_effective - 1) // 2
                    pad_w_end = kernel_width_effective - 1 - pad_w_beg
                    padding = [[0, 0], [pad_h_beg, pad_h_end],
                               [pad_w_beg, pad_w_end], [0, 0]]
                    if self.data_format == 'NCHW':
                        padding = [padding[0], padding[3], padding[1], padding[2]]
                    input_layer = tf.pad(input_layer, padding)
                    conv = self._conv2d_impl(input_layer, num_channels_in,
                        num_out_channels,
                        kernel_size=[k_height, k_width],
                        strides=[d_height, d_width], padding='VALID',
                        kernel_initializer=kernel_initializer)
                    # assert False, 'shawn does not know what this is'

            if use_batch_norm is None:
                use_batch_norm = self.use_batch_norm
            if not use_batch_norm:
                if bias is not None:
                    biases = self.get_variable('biases', [num_out_channels],
                        self.variable_dtype, self.dtype,
                        initializer=tf.constant_initializer(bias))
                    biased = tf.reshape(
                        tf.nn.bias_add(conv, biases, data_format=self.data_format),
                        conv.get_shape())
                else:
                    biased = conv
            else:
                self.top_layer = conv
                self.top_size = num_out_channels
                biased = self.batch_norm(**self.batch_norm_config)

            #   TODO record the internal outputs of conv
            if self.need_record_internal_outputs:
                if len(self.internal_outputs_dict) == 0:
                    self.internal_outputs_dict['input'] = tf.identity(input_layer)
                self.internal_outputs_dict['{}${}'.format(self.num_internal_conv_outputs, name)] = conv
                self.internal_outputs_dict['{}#{}'.format(self.num_internal_conv_outputs, name)] = biased
                self.num_internal_conv_outputs += 1

            if activation == 'relu':
                conv1 = tf.nn.relu(biased)
            elif activation == 'linear' or activation is None:
                conv1 = biased
            elif activation == 'tanh':
                conv1 = tf.nn.tanh(biased)
            elif activation == 'sigmoid':
                conv1 = tf.nn.sigmoid(biased)
            else:
                raise KeyError('Invalid activation type \'%s\'' % activation)
            self.top_layer = conv1
            self.top_size = num_out_channels
            return conv1

    def se_rescale(self, input, internal_neurons, specify_name=True):
        if self.data_format == 'NHWC':
            pooled_inputs = tf.reduce_mean(input, [1, 2], keep_dims=True)
            num_channels = input.get_shape().as_list()[3]
        else:
            pooled_inputs = tf.reduce_mean(input, [2, 3], keep_dims=True)
            num_channels = input.get_shape().as_list()[1]
        if specify_name:
            up_name = 'seup'
            down_name = 'sedown'
            up_name_postfix = None
            down_name_postfix = None
            count_convs = False
        else:
            up_name = None
            down_name = None
            up_name_postfix = '_seup'
            down_name_postfix = '_sedown'
            count_convs = True
        #TODO just_as_classic_conv is used to prevent other convnetbuilders from changing the conv implementation. is there any better solution?
        down_inputs = self.conv(internal_neurons, 1, 1, 1, 1,  input_layer=pooled_inputs, activation='relu', mode='VALID',
            use_batch_norm=False, bias=0.0, name=down_name, name_postfix=down_name_postfix, count_convs=count_convs, just_as_classic_conv=True)
        up_inputs = self.conv(num_channels, 1, 1, 1, 1, input_layer=down_inputs, activation=None, mode='VALID',
            use_batch_norm=False, bias=0.0, name=up_name, name_postfix=up_name_postfix, count_convs=count_convs, just_as_classic_conv=True)
        prob_outputs = tf.nn.sigmoid(up_inputs)
        rescaled = tf.multiply(prob_outputs, input)
        return rescaled


    def _pool(self,
              pool_name,
              pool_function,
              k_height,
              k_width,
              d_height,
              d_width,
              mode,
              input_layer,
              num_channels_in):
        """Construct a pooling layer."""
        if input_layer is None:
            input_layer = self.top_layer
        else:
            self.top_size = num_channels_in
        name = pool_name + str(self.counts[pool_name])
        self.counts[pool_name] += 1
        if self.use_tf_layers:
            pool = pool_function(
                input_layer, [k_height, k_width], [d_height, d_width],
                padding=mode,
                data_format=self.channel_pos,
                name=name)
        else:
            if self.data_format == 'NHWC':
                ksize = [1, k_height, k_width, 1]
                strides = [1, d_height, d_width, 1]
            else:
                ksize = [1, 1, k_height, k_width]
                strides = [1, 1, d_height, d_width]
            pool = tf.nn.max_pool(input_layer, ksize, strides, padding=mode,
                data_format=self.data_format, name=name)
        self.top_layer = pool
        return pool

    def mpool(self,
              k_height,
              k_width,
              d_height=2,
              d_width=2,
              mode='VALID',
              input_layer=None,
              num_channels_in=None):
        """Construct a max pooling layer."""
        return self._pool('mpool', pooling_layers.max_pooling2d, k_height, k_width,
            d_height, d_width, mode, input_layer, num_channels_in)

    def apool(self,
              k_height,
              k_width,
              d_height=2,
              d_width=2,
              mode='VALID',
              input_layer=None,
              num_channels_in=None):
        """Construct an average pooling layer."""
        return self._pool('apool', pooling_layers.average_pooling2d, k_height,
            k_width, d_height, d_width, mode, input_layer,
            num_channels_in)

    def reshape(self, shape, input_layer=None):
        if input_layer is None:
            input_layer = self.top_layer
        self.top_layer = tf.reshape(input_layer, shape)
        self.top_size = shape[-1]  # HACK This may not always work
        return self.top_layer

    def flatten(self, input_layer=None):
        if input_layer is None:
            input_layer = self.top_layer
        shape = input_layer.get_shape().as_list()
        flat_dim = shape[1] * shape[2] * shape[3]
        self.top_layer = self.reshape([-1, flat_dim], input_layer=input_layer)
        return self.top_layer

    def affine(self,
               num_out_channels,
               input_layer=None,
               num_channels_in=None,
               bias=0.0,
               stddev=None,
               activation='relu'):
        if input_layer is None:
            input_layer = self.top_layer
        if num_channels_in is None:
            num_channels_in = self.top_size
        name = 'affine' + str(self.counts['affine'])
        self.counts['affine'] += 1
        with tf.variable_scope(name):
            if not self.use_dense_layer:
                init_factor = 2. if activation == 'relu' else 1.
                stddev = stddev or np.sqrt(init_factor / num_channels_in)
                kernel = self.get_variable(
                    'weights', [num_channels_in, num_out_channels],
                    self.variable_dtype, self.dtype,
                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                biases = self.get_variable('biases', [num_out_channels],
                    self.variable_dtype, self.dtype,
                    initializer=tf.constant_initializer(bias))
                logits = tf.nn.xw_plus_b(input_layer, kernel, biases)
            else:
                print('use tf dense layer!!!')
                logits = dense(input_layer, num_out_channels, activation=None, use_bias=True,
                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            if activation == 'relu':
                affine1 = tf.nn.relu(logits, name=name)
            elif activation == 'linear' or activation is None:
                affine1 = logits
            else:
                raise KeyError('Invalid activation type \'%s\'' % activation)
            self.top_layer = affine1
            self.top_size = num_out_channels
            return affine1

    def inception_module(self, name, cols, input_layer=None, in_size=None):
        if input_layer is None:
            input_layer = self.top_layer
        if in_size is None:
            in_size = self.top_size
        name += str(self.counts[name])
        self.counts[name] += 1
        with tf.variable_scope(name):
            col_layers = []
            col_layer_sizes = []
            for c, col in enumerate(cols):
                col_layers.append([])
                col_layer_sizes.append([])
                for l, layer in enumerate(col):
                    ltype, args = layer[0], layer[1:]
                    kwargs = {
                        'input_layer': input_layer,
                        'num_channels_in': in_size
                    } if l == 0 else {}
                    if ltype == 'conv':
                        self.conv(*args, **kwargs)
                    elif ltype == 'mpool':
                        self.mpool(*args, **kwargs)
                    elif ltype == 'apool':
                        self.apool(*args, **kwargs)
                    elif ltype == 'share':  # Share matching layer from previous column
                        self.top_layer = col_layers[c - 1][l]
                        self.top_size = col_layer_sizes[c - 1][l]
                    else:
                        raise KeyError(
                            'Invalid layer type for inception module: \'%s\'' % ltype)
                    col_layers[c].append(self.top_layer)
                    col_layer_sizes[c].append(self.top_size)
            catdim = 3 if self.data_format == 'NHWC' else 1
            self.top_layer = tf.concat([layers[-1] for layers in col_layers], catdim)
            self.top_size = sum([sizes[-1] for sizes in col_layer_sizes])
            return self.top_layer

    def dropout(self, keep_prob=0.5, input_layer=None):
        if input_layer is None:
            input_layer = self.top_layer
        else:
            self.top_size = None
        name = 'dropout' + str(self.counts['dropout'])
        with tf.variable_scope(name):
            if not self.phase_train:
                keep_prob = 1.0
            if self.use_tf_layers:
                dropout = core_layers.dropout(input_layer, 1. - keep_prob,
                    training=self.phase_train)
            else:
                dropout = tf.nn.dropout(input_layer, keep_prob)
            self.top_layer = dropout
            return dropout

    def _batch_norm_without_layers(self, input_layer, decay, use_scale, epsilon):
        """Batch normalization on `input_layer` without tf.layers."""
        # We make this function as similar as possible to the
        # tf.contrib.layers.batch_norm, to minimize the differences between using
        # layers and not using layers.
        shape = input_layer.shape
        num_channels = shape[3] if self.data_format == 'NHWC' else shape[1]
        beta = self.get_variable('beta', [num_channels], tf.float32, tf.float32,
            initializer=tf.zeros_initializer())
        if use_scale:
            gamma = self.get_variable('gamma', [num_channels], tf.float32,
                tf.float32, initializer=tf.ones_initializer())
        else:
            gamma = tf.constant(1.0, tf.float32, [num_channels])
        # For moving variables, we use tf.get_variable instead of self.get_variable,
        # since self.get_variable returns the result of tf.cast which we cannot
        # assign to.
        moving_mean = tf.get_variable('moving_mean', [num_channels],
            tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=False)
        moving_variance = tf.get_variable('moving_variance', [num_channels],
            tf.float32,
            initializer=tf.ones_initializer(),
            trainable=False)
        if self.phase_train:
            bn, batch_mean, batch_variance = tf.nn.fused_batch_norm(
                input_layer, gamma, beta, epsilon=epsilon,
                data_format=self.data_format, is_training=True)
            mean_update = moving_averages.assign_moving_average(
                moving_mean, batch_mean, decay=decay, zero_debias=False)
            variance_update = moving_averages.assign_moving_average(
                moving_variance, batch_variance, decay=decay, zero_debias=False)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_update)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, variance_update)
        else:
            bn, _, _ = tf.nn.fused_batch_norm(
                input_layer, gamma, beta, mean=moving_mean,
                variance=moving_variance, epsilon=epsilon,
                data_format=self.data_format, is_training=False)
        return bn


    def batch_norm(self, input_layer=None, decay=None, scale=None,
                   epsilon=None, for_shadow=False, name=None, center=True, gamma_init=1):
        """Adds a Batch Normalization layer."""
        if input_layer is None:
            input_layer = self.top_layer
        else:
            self.top_size = None
        if decay is None:
            decay = self.batch_norm_config['decay']
        if scale is None:
            scale = self.batch_norm_config['scale']
        if epsilon is None:
            epsilon = self.batch_norm_config['epsilon']

        if name is None:
            name = 'batchnorm' + str(self.counts['batchnorm'])
        self.counts['batchnorm'] += 1

        assert gamma_init in [0, 1]
        if gamma_init == 1:
            param_initializers = None
        else:
            param_initializers = {'gamma': tf.zeros_initializer()}
            print('the initializer for gamma ', param_initializers)

        with tf.variable_scope(name) as scope:
            if self.use_tf_layers:
                bn = tf.contrib.layers.batch_norm(
                    input_layer,
                    decay=decay,
                    center=center,
                    scale=scale,
                    epsilon=epsilon,
                    is_training=self.phase_train,
                    fused=True,
                    data_format=self.data_format,
                    scope=scope,
                    param_initializers=param_initializers)
            else:
                bn = self._batch_norm_without_layers(input_layer, decay, scale, epsilon)
        self.top_layer = bn
        self.top_size = bn.shape[3] if self.data_format == 'NHWC' else bn.shape[1]
        self.top_size = int(self.top_size)
        return bn

    def lrn(self, depth_radius, bias, alpha, beta):
        """Adds a local response normalization layer."""
        name = 'lrn' + str(self.counts['lrn'])
        self.counts['lrn'] += 1
        self.top_layer = tf.nn.lrn(
            self.top_layer, depth_radius, bias, alpha, beta, name=name)
        return self.top_layer
