from convnet_builder import *
import tensorflow as tf


class BDSConvNetBuilder(ConvNetBuilder):
    def __init__(self,
                 input_op,
                 input_nchan,
                 phase_train,
                 use_tf_layers,
                 data_format,
                 bds_params,
                 dtype=tf.float32,
                 variable_dtype=tf.float32,
                 use_dense_layer=False):
        super(BDSConvNetBuilder, self).__init__(input_op=input_op, input_nchan=input_nchan, phase_train=phase_train,
            use_tf_layers=use_tf_layers, data_format=data_format, dtype=dtype, variable_dtype=variable_dtype, use_dense_layer=use_dense_layer)
        self.bds_params = bds_params
        assert self.bds_params.metric_type in ['euc']
        self.cur_layer_idx = 0
        self.based = {}
        self.dsed = {}
        self.normal_output = {}
        self.shadow_output = {}
        self.base_masks = {}
        self.ds_masks = {}
        self.first_fc_normal_output = None
        self.first_fc_shadow_output = None

    def _pool(self,
              pool_name,
              pool_function,
              k_height,
              k_width,
              d_height,
              d_width,
              mode,
              input_layer,
              num_channels_in,
              for_shadow=False):
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

        if not for_shadow:
            self.based[self.cur_layer_idx - 1] = pool
            self.top_layer = pool
            if ((self.cur_layer_idx - 1) in self.bds_params.target_layers):
                self.dsed[self.cur_layer_idx - 1] = self._pool(pool_name=pool_name, pool_function=pool_function,
                    k_height=k_height, k_width=k_width, d_height=d_height, d_width=d_width, mode=mode,
                    input_layer=self.dsed[self.cur_layer_idx - 1], num_channels_in=num_channels_in, for_shadow=True)

        return pool

    def batch_norm(self, input_layer=None, decay=0.9, scale=True,
                   epsilon=0.001, for_shadow=False):

        """Adds a Batch Normalization layer."""
        assert input_layer is not None

        if for_shadow:
            name = 'batchnorm' + str(self.counts['batchnorm'] - 1)
        else:
            name = 'batchnorm' + str(self.counts['batchnorm'])
            self.counts['batchnorm'] += 1

        with tf.variable_scope(name, reuse=for_shadow) as scope:
            if self.use_tf_layers:
                bn = tf.contrib.layers.batch_norm(
                    input_layer,
                    decay=decay,
                    scale=scale,
                    epsilon=epsilon,
                    is_training=self.phase_train,
                    fused=True,
                    data_format=self.data_format,
                    scope=scope)
            else:
                bn = self._batch_norm_without_layers(input_layer, decay, scale, epsilon)

        # if not for_shadow:
        #     self.top_layer = bn
        #     self.top_size = bn.shape[3] if self.data_format == 'NHWC' else bn.shape[1]
        #     self.top_size = int(self.top_size)
        return bn

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
             for_shadow=False,
             specify_padding=None):
        """Construct a conv2d layer on top of cnn."""
        if input_layer is None:
            input_layer = self.top_layer
        if num_channels_in is None:
            num_channels_in = self.top_size
        if stddev is not None and kernel_initializer is None:
            kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)


        if not for_shadow:
            name = 'conv' + str(self.counts['conv'])
            self.counts['conv'] += 1
        else:
            name = 'conv' + str(self.counts['conv'] - 1)

        with tf.variable_scope(name, reuse=for_shadow):
            strides = [1, d_height, d_width, 1]
            if self.data_format == 'NCHW':
                strides = [strides[0], strides[3], strides[1], strides[2]]
            if mode != 'SAME_RESNET':
                conv = self._conv2d_impl(input_layer, num_channels_in, num_out_channels,
                    kernel_size=[k_height, k_width],
                    strides=[d_height, d_width], padding=mode,
                    kernel_initializer=kernel_initializer,
                    specify_padding=specify_padding)
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
                biased = self.batch_norm(**self.batch_norm_config, input_layer=conv, for_shadow=for_shadow)

            if activation == 'relu':
                conv1 = tf.nn.relu(biased)
            elif activation == 'linear' or activation is None:
                conv1 = biased
            elif activation == 'tanh':
                conv1 = tf.nn.tanh(biased)
            else:
                raise KeyError('Invalid activation type \'%s\'' % activation)

        if for_shadow:
            return conv1


        if self.cur_layer_idx in self.bds_params.target_layers:

            filter_pos = 3 if self.data_format == 'NHWC' else 1

            base_ph = tf.placeholder(dtype=tf.float32, shape=(conv1.get_shape()[filter_pos]))
            self.base_masks[self.cur_layer_idx] = base_ph

            ds_ph = tf.placeholder(dtype=tf.float32, shape=(conv1.get_shape()[filter_pos],))
            self.ds_masks[self.cur_layer_idx] = ds_ph

            print('the base mask named {} shape {}'.format(base_ph.name, base_ph.get_shape()))

            conv1_shape = [v.value for v in conv1.get_shape()]
            if self.data_format == 'NHWC':
                print('tile shape:', [conv1_shape[0], conv1_shape[1], conv1_shape[2], 1])
                expanded_base = tf.reshape(base_ph, (1, 1, 1, conv1_shape[3]))
                tiled_base_ph = tf.manip.tile(expanded_base, [conv1_shape[0], conv1_shape[1], conv1_shape[2], 1],
                                              name='base_mask_layer{}'.format(self.cur_layer_idx))
                expanded_ds = tf.reshape(ds_ph, (1, 1, 1, conv1_shape[3]))
                tiled_ds_ph = tf.manip.tile(expanded_ds, [conv1_shape[0], conv1_shape[1], conv1_shape[2], 1],
                                            name='ds_mask_layer{}'.format(self.cur_layer_idx))
            else:
                assert self.data_format == 'NCHW'
                print('tile shape:', [conv1_shape[0], 1, conv1_shape[2], conv1_shape[3]])
                expanded_base = tf.reshape(base_ph, (1, conv1_shape[1], 1, 1))
                tiled_base_ph = tf.manip.tile(expanded_base, [conv1_shape[0], 1, conv1_shape[2], conv1_shape[3]],
                                              name='base_mask_layer{}'.format(self.cur_layer_idx))
                expanded_ds = tf.reshape(ds_ph, (1, conv1_shape[1], 1, 1))
                tiled_ds_ph = tf.manip.tile(expanded_ds, [conv1_shape[0], 1, conv1_shape[2], conv1_shape[3]],
                                            name='ds_mask_layer{}'.format(self.cur_layer_idx))
            self.dsed[self.cur_layer_idx] = conv1 * tiled_ds_ph
            self.based[self.cur_layer_idx] = conv1 * tiled_base_ph
            self.top_layer = self.based[self.cur_layer_idx]
        else:
            self.top_layer = conv1

        self.normal_output[self.cur_layer_idx] = conv1
        self.top_size = num_out_channels

        if self.cur_layer_idx > 0 and ((self.cur_layer_idx - 1) in self.bds_params.target_layers):

            self.shadow_output[self.cur_layer_idx] = self.conv(num_out_channels=num_out_channels, k_height=k_height,
                k_width=k_width, d_height=d_height, d_width=d_width, mode=mode,
                input_layer=self.dsed[self.cur_layer_idx - 1],  # TODO may not work for resnets
                num_channels_in=num_channels_in, use_batch_norm=use_batch_norm, stddev=stddev,
                activation=activation,
                bias=bias, kernel_initializer=kernel_initializer, for_shadow=True, specify_padding=specify_padding)

        self.cur_layer_idx += 1

        return conv1


    def affine(self,
               num_out_channels,
               input_layer=None,
               num_channels_in=None,
               bias=0.0,
               stddev=None,
               activation='relu',
               for_shadow=False):
        if input_layer is None:
            input_layer = self.top_layer
        if num_channels_in is None:
            num_channels_in = self.top_size

        if not for_shadow:
            name = 'affine' + str(self.counts['affine'])
            self.counts['affine'] += 1
        else:
            name = 'affine' + str(self.counts['affine'] - 1)

        with tf.variable_scope(name, reuse=for_shadow):
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

            if not for_shadow:
                self.top_layer = affine1
                self.top_size = num_out_channels

        if (not for_shadow) and self.counts['affine'] == 1:  # this is the first fc
            assert self.first_fc_normal_output is None
            assert self.first_fc_shadow_output is None
            # assert self.cur_layer_idx - 1 == max(self.dsed.keys())
            self.first_fc_normal_output = affine1
            if ((self.cur_layer_idx - 1) in self.bds_params.target_layers):
                last_conv_dsed = self.dsed[self.cur_layer_idx - 1]
                self.first_fc_shadow_output = self.affine(num_out_channels=num_out_channels,
                    input_layer=last_conv_dsed, num_channels_in=num_channels_in, bias=bias, stddev=stddev,
                    activation=activation, for_shadow=True)

        return affine1


    def reshape(self, shape, input_layer=None):
        if input_layer is None:
            input_layer = self.top_layer
        self.top_layer = tf.reshape(input_layer, shape)
        self.top_size = shape[-1]  # HACK This may not always work

        if ((self.cur_layer_idx - 1) in self.bds_params.target_layers):
            self.dsed[self.cur_layer_idx - 1] = tf.reshape(self.dsed[self.cur_layer_idx - 1], shape)

        return self.top_layer


    def get_base_masks(self):
        return self.base_masks

    def get_ds_masks(self):
        return self.ds_masks

    def get_bds_metrics(self):
        print('the current metric is ', self.bds_params.metric_type)
        print('len of normal_output is {}, len of shadow_output is {}'.format(len(self.normal_output), len(self.shadow_output)))
        result = {}
        for i in self.shadow_output.keys():
            result[i - 1] = tf.reduce_mean((self.normal_output[i] - self.shadow_output[i]) ** 2) / tf.reduce_mean(self.normal_output[i] ** 2)        #   TODO may not work for resnets
        if (self.cur_layer_idx - 1) in self.bds_params.target_layers:
            result[self.cur_layer_idx - 1] = tf.reduce_mean((self.first_fc_shadow_output - self.first_fc_normal_output) ** 2) / tf.reduce_mean(self.first_fc_normal_output ** 2)
        print('the keys of bds metrics are {}'.format(result.keys()))
        return result
