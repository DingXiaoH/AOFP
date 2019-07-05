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
"""TensorFlow benchmark library.

See the README for more information.
"""

from __future__ import print_function


from bc_helpers import *
from tf_utils import *
from bc_util import *


_DEFAULT_NUM_BATCHES = 100



class BenchmarkBase(object):
    """Class for benchmarking a cnn network."""

    #   shawn
    #   force_subset:   if we wish to test the model on the training set, set force_subset='train'
    #   note:   self.batch_size refers to the global batchsize (batch_size_per_gpu * num_gpus)
    #           but self.num_epochs and self.num_batches (which we use as the termination conditions) are divided by num_workers (for distributed training)
    #           self.benchmark_logger is None as default
    def __init__(self, params, my_params, dataset=None, model=None, ):
        """Initialize BenchmarkCNN.

        Args:
          params: Params tuple, typically created by make_params or
                  make_params_from_flags.
          dataset: If not None, the dataset to use. Otherwise, params is used to
                   obtain the dataset.
          model: If not None, the model to use. Otherwise, params is used to obtain
                 the model.
        Raises:
          ValueError: Unsupported params settings.
        """
        self.params = params
        self.my_params = my_params


        self.dataset = dataset or datasets.create_dataset(self.params.data_dir,
            self.params.data_name)
        self.model = model or model_config.get_model_config(
            self.params.model, self.dataset, self.params)
        self.trace_filename = self.params.trace_file
        self.rewriter_config = self.params.rewriter_config
        autotune_threshold = self.params.autotune_threshold if (
            self.params.autotune_threshold) else 1
        min_autotune_warmup = 5 * autotune_threshold * autotune_threshold
        self.num_warmup_batches = self.params.num_warmup_batches if (
            self.params.num_warmup_batches is not None) else max(
            10, min_autotune_warmup)
        self.graph_file = self.params.graph_file
        self.resize_method = self.params.resize_method
        self.sync_queue_counter = 0
        self.num_gpus = self.params.num_gpus
        if self.params.gpu_indices:
            self.gpu_indices = [int(x) for x in self.params.gpu_indices.split(',')]
        else:
            self.gpu_indices = [x for x in range(self.num_gpus)]

        if (self.params.device == 'cpu' and self.params.data_format == 'NCHW' and
                not self.params.mkl):
            raise ValueError('device=cpu requires that data_format=NHWC')

        if ((self.params.num_epochs_per_decay or
                 self.params.learning_rate_decay_factor) and
                not (self.params.init_learning_rate and self.params.num_epochs_per_decay
                     and self.params.learning_rate_decay_factor)):
            raise ValueError('If one of num_epochs_per_decay or '
                             'learning_rate_decay_factor is set, both must be set'
                             'and learning_rate must be set')
        if (self.params.minimum_learning_rate and
                not (self.params.init_learning_rate and self.params.num_epochs_per_decay
                     and self.params.learning_rate_decay_factor)):
            raise ValueError('minimum_learning_rate requires learning_rate,'
                             'num_epochs_per_decay, and '
                             'learning_rate_decay_factor to be set')

        if (self.params.use_fp16 and self.params.fp16_vars and
                    'replicated' in self.params.variable_update and
                self.params.all_reduce_spec and 'nccl' in self.params.all_reduce_spec):
            raise ValueError('fp16 variables are not supported with NCCL')
        if (self.params.use_fp16 and self.params.fp16_vars and
                self.params.gradient_repacking):
            raise ValueError('--fp16_vars cannot be used with --gradient_repacking')

        if self.params.variable_update == 'horovod' and self.params.num_gpus > 1:
            raise ValueError('Horovod benchmarks require num_gpus=1 on each worker')

        if self.params.variable_update == 'horovod' and self.params.job_name:
            raise ValueError('job_name should not be specified for Horovod.')

        if self.params.use_fp16 and self.params.fp16_enable_auto_loss_scale:
            if self.params.all_reduce_spec and 'nccl' in self.params.all_reduce_spec:
                raise ValueError('Automatic loss scaling is not supported with NCCL.')
            if self.params.variable_update not in ('parameter_server', 'replicated',
                                                   'independent'):
                raise ValueError('Automatic loss scaling is not supported with '
                                 'variable_update=%s.' % self.params.variable_update)
            if self.params.staged_vars:
                raise ValueError('Automatic loss scaling is not supported with'
                                 'staged_vars.')

        if (self.params.debugger is not None and self.params.debugger != 'cli' and
                    ':' not in self.params.debugger):
            raise ValueError('--debugger must be "cli" or in the form '
                             'host:port')

        if self.params.hierarchical_copy and self.params.num_gpus <= 1:
            raise ValueError('--hierarchical_copy requires --num_gpus to be greater '
                             'than 1')

        if params.save_model_secs and params.save_model_steps:
            raise ValueError('At most one of --save_model_secs and '
                             '--save_model_steps can be specified')

        if self.params.forward_only and self.params.freeze_when_forward_only:
            if self.params.train_dir is not None:
                raise ValueError('In forward_only mode, when --freeze_when_forward_only'
                                 ' is True, --train_dir should not be specified')
            if self.params.data_dir and not self.params.datasets_use_prefetch:
                raise ValueError('In forward_only mode, when --freeze_when_forward_only'
                                 ' is True and --data_dir is set, '
                                 '--datasets_use_prefetch should be set to True')
            if self.params.job_name:
                raise ValueError('In forward_only mode, when --freeze_when_forward_only'
                                 ' is True, --job_name should not be specified and '
                                 'distributed running is not supported')
            self.forward_only_and_freeze = True
        else:
            self.forward_only_and_freeze = False
            if self.params.trt_mode:
                raise ValueError('--trt_mode should not be specified if one of '
                                 '--forward_only and --freeze_when_forward_only is set '
                                 'to False')

        if not self.params.datasets_use_prefetch:
            if self.params.datasets_use_caching:
                raise ValueError('Dataset caching is only supported for '
                                 '--datasets_use_prefetch=True')
            if self.params.datasets_parallel_interleave_cycle_length is not None:
                raise ValueError('Setting parallel interleave cycle length is only '
                                 'supported for --datasets_use_prefetch=True')
            if self.params.datasets_sloppy_parallel_interleave:
                raise ValueError('Sloppy parallel interleave is only supported for '
                                 '--datasets_use_prefetch=True')

        # Use the batch size from the command line if specified, otherwise use the
        # model's default batch size.  Scale the benchmark's batch size by the
        # number of GPUs.
        if self.params.batch_size > 0:
            self.model.set_batch_size(self.params.batch_size)
        self.batch_size = self.model.get_batch_size() * self.num_gpus
        self.batch_group_size = self.params.batch_group_size
        self.enable_auto_loss_scale = (
            self.params.use_fp16 and self.params.fp16_enable_auto_loss_scale)
        self.loss_scale = None
        self.loss_scale_normal_steps = None

        self.job_name = self.params.job_name  # "" for local training

        # PS server is used for distributed jobs not using all-reduce.
        use_ps_server = self.job_name and (self.params.variable_update !=
                                           'distributed_all_reduce' and
                                           self.params.variable_update !=
                                           'collective_all_reduce')
        # controller is used for distributed_all_reduce with > 1 worker.
        use_controller = (
            self.params.variable_update == 'distributed_all_reduce' and
            self.job_name)
        if use_controller and not params.controller_host:
            raise ValueError('When variable_update==distributed_all_reduce '
                             'controller_host must also be specified.')
        # collective_all_reduce doesn't need a controller or ps
        self.distributed_collective = (
            self.params.variable_update == 'collective_all_reduce' and
            self.job_name)

        self.local_parameter_device_flag = self.params.local_parameter_device
        if self.job_name:
            self.task_index = self.params.task_index
            self.cluster_manager = platforms_util.get_cluster_manager(
                params, create_config_proto(params))
            assert isinstance(self.cluster_manager, cnn_util.BaseClusterManager)

            worker_prefix = '/job:worker/replica:0/task:%s' % self.task_index
            if use_ps_server:
                self.param_server_device = tf.train.replica_device_setter(
                    worker_device=worker_prefix + '/cpu:0',
                    cluster=self.cluster_manager.get_cluster_spec())
                # This device on which the queues for managing synchronization between
                # servers should be stored.
                self.sync_queue_devices = [
                    '/job:ps/replica:0/task:%s/cpu:0' % i
                    for i in range(self.cluster_manager.num_ps())
                ]
            else:
                self.sync_queue_devices = ['/job:worker/replica:0/task:0/cpu:0']
        else:
            self.task_index = 0
            self.cluster_manager = None
            worker_prefix = ''
            self.param_server_device = '/%s:0' % self.params.local_parameter_device
            self.sync_queue_devices = [self.param_server_device]

        if self.cluster_manager:
            self.num_workers = self.cluster_manager.num_workers()
        elif self.params.variable_update == 'horovod':
            import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
            self.num_workers = hvd.size()
        else:
            self.num_workers = 1
        self.num_ps = self.cluster_manager.num_ps() if self.cluster_manager else 0

        if self.num_workers > 1 and self.params.all_reduce_spec == 'nccl':
            raise ValueError('--all_reduce_spec=nccl is invalid in a '
                             'multi-worker job')

        # Device to use for ops that need to always run on the local worker's CPU.
        self.cpu_device = '%s/cpu:0' % worker_prefix

        # Device to use for ops that need to always run on the local worker's
        # compute device, and never on a parameter server device.
        self.raw_devices = [
            '%s/%s:%i' % (worker_prefix, self.params.device, i)
            for i in xrange(self.num_gpus)
        ]

        self.subset = self.my_params.force_subset or ('validation' if params.eval else 'train')

        self.num_batches, self.num_epochs = get_num_batches_and_epochs(
            params, self.batch_size * self.num_workers,
            self.dataset.num_examples_per_epoch(self.subset))

        if (self.params.staged_vars and
                    self.params.variable_update != 'parameter_server'):
            raise ValueError('staged_vars for now is only supported with '
                             'variable_update=parameter_server')

        if self.params.variable_update == 'parameter_server':
            if self.job_name:
                if not self.params.staged_vars:
                    self.variable_mgr = variable_mgr.VariableMgrDistributedFetchFromPS(
                        self)
                else:
                    self.variable_mgr = (
                        variable_mgr.VariableMgrDistributedFetchFromStagedPS(self))
            else:
                if not self.params.staged_vars:
                    self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
                else:
                    self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromStagedPS(
                        self)
        elif self.params.variable_update == 'replicated':
            print('using replicated training')
            if self.job_name:
                raise ValueError('Invalid variable_update in distributed mode: %s' %
                                 self.params.variable_update)
            self.variable_mgr = variable_mgr.VariableMgrLocalReplicated(
                self, self.params.all_reduce_spec,
                self.params.agg_small_grads_max_bytes,
                self.params.agg_small_grads_max_group,
                self.params.allreduce_merge_scope)
        elif self.params.variable_update == 'distributed_all_reduce':
            assert self.params.cross_replica_sync
            self.variable_mgr = variable_mgr.VariableMgrDistributedAllReduce(
                self, self.params.all_reduce_spec,
                ('worker' if self.num_workers > 1 else 'localhost'),
                self.num_workers, self.params.agg_small_grads_max_bytes,
                self.params.agg_small_grads_max_group,
                self.params.allreduce_merge_scope)
        elif self.params.variable_update == 'collective_all_reduce':
            assert self.params.cross_replica_sync
            self.variable_mgr = variable_mgr.VariableMgrCollectiveAllReduce(
                self, self.params.all_reduce_spec,
                self.num_workers, self.num_gpus, self.task_index,
                self.params.allreduce_merge_scope)
        elif self.params.variable_update == 'distributed_replicated':
            assert self.params.cross_replica_sync
            if not self.job_name:
                raise ValueError('Invalid variable_update in local mode: %s' %
                                 self.params.variable_update)
            self.variable_mgr = variable_mgr.VariableMgrDistributedReplicated(self)
        elif self.params.variable_update in ('independent', 'horovod'):
            if self.job_name:
                raise ValueError('Invalid variable_update in distributed mode: %s' %
                                 self.params.variable_update)
            self.variable_mgr = variable_mgr.VariableMgrIndependent(self)
        else:
            raise ValueError(
                'Invalid variable_update: %s' % self.params.variable_update)

        # Device to use for running on the local worker's compute device, but
        # with variables assigned to parameter server devices.
        self.devices = self.variable_mgr.get_devices()
        if self.job_name:
            if use_ps_server:
                self.global_step_device = self.param_server_device
            elif self.params.variable_update == 'collective_all_reduce':
                self.global_step_device = self.cpu_device
            else:
                self.global_step_device = '/job:worker/replica:0/task:0/cpu:0'
        else:
            self.global_step_device = self.cpu_device

        self.input_preprocessor = None
        if not self.dataset.use_synthetic_gpu_inputs():
            self.input_preprocessor = self.get_input_preprocessor()
        self.datasets_use_prefetch = (
            self.params.datasets_use_prefetch and
            # TODO(rohanj): Figure out why --datasets_use_prefetch freezes on the
            # CPU.
            self.params.device.lower() != 'cpu' and
            self.input_preprocessor and
            self.input_preprocessor.supports_datasets())
        self.init_global_step = 0

        self._config_benchmark_logger()

        self.gradient_handler = None
        self.sess = None
        self.graph = None
        self.name_to_variables = None


    def set_gradient_handler(self, gh):
        self.gradient_handler = gh


    def get_global_variables(self):
        return self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def update_name_to_variables(self):
        self.name_to_variables = {t.name : t for t in self.get_global_variables()}


    def _get_variables_by_keyword(self, keyword):
        result = []
        if keyword in ['moving_mean', 'moving_variance']:
            v_collect = self.get_global_variables()
        else:
            v_collect = self.get_trainable_variables()
        for t in v_collect:
            if keyword in t.name:
                result.append(t)
        return result

    def get_trainable_variables(self):
        return self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def get_kernel_variables(self, target_layers=None):
        result = []
        for t in self.get_key_variables():
            if ('kernel' in t.name or 'weights' in t.name) and ('seup' not in t.name and 'sedown' not in t.name):
                result.append(t)
        if target_layers is not None:
            result = [result[i] for i in target_layers]
            for r in result:
                assert r.name.startswith('v0')
        return result

    def get_bias_variables(self):
        return self._get_variables_by_keyword('bias')

    def get_moving_mean_variables(self):
        return self._get_variables_by_keyword('moving_mean')

    def get_moving_variance_variables(self):
        return self._get_variables_by_keyword('moving_variance')

    def get_gamma_variables(self):
        return self._get_variables_by_keyword('gamma')

    def get_beta_variables(self):
        return self._get_variables_by_keyword('beta')

    def get_lmd_variables(self):
        return self._get_variables_by_keyword('lmd')

    def get_phi_variables(self):
        return self._get_variables_by_keyword('phi')

    # return a dict
    def get_variable_values(self, variables):
        var_v = []
        var_names = []
        for v in variables:
            if v is not None:
                var_v.append(v)
                var_names.append(v.name)
        var_values = self.get_value(var_v)
        result = {}
        for n, v in zip(var_names, var_values):
            result[n] = v
        return result

    #   deprecated
    # def set_variable_values(self, variables, values):
    #     cnt = 0
    #     for v in variables:
    #         if v.name in values:
    #             self.set_value(v, values[v.name])
    #             cnt += 1
    #     print('set values for {} variables'.format(cnt))

    def _get_variable_for_kernel(self, kernel_var, keyword):
        'v0/cg/conv1/kernel:0'
        kernel_name = kernel_var.name
        if 'kernel' in kernel_name:
            variable_name = kernel_name.replace('kernel', keyword)
        else:
            assert 'weights' in kernel_name
            variable_name = kernel_name.replace('weights', keyword)
        # variable_name = variable_name.replace('_x1/', '_x2/')       #TODO ugly hack, for DenseNet only
        if variable_name in self.name_to_variables:
            return self.name_to_variables[variable_name]
        else:
            possible_name = variable_name.replace('/' + keyword, '_bn/' + keyword)  #TODO ugly hack again
            if possible_name in self.name_to_variables:
                return self.name_to_variables[possible_name]
            else:
                if keyword == 'bias':
                    possible_name = kernel_name.replace('conv2d/kernel', 'biases')  #TODO ugly hack again and again
                    if possible_name in self.name_to_variables:
                        return self.name_to_variables[possible_name]
                return try_to_get_variable_by_eliminating_batchnorm_words(self.name_to_variables, kernel_name.replace('conv2d/kernel', keyword))

    def get_bias_variable_for_kernel(self, kernel_var):
        return self._get_variable_for_kernel(kernel_var, 'bias')

    def get_moving_mean_variable_for_kernel(self, kernel_var):
        return self._get_variable_for_kernel(kernel_var, 'moving_mean')

    def get_moving_variance_variable_for_kernel(self, kernel_var):
        return self._get_variable_for_kernel(kernel_var, 'moving_variance')

    def get_beta_variable_for_kernel(self, kernel_var):
        return self._get_variable_for_kernel(kernel_var, 'beta')

    def get_gamma_variable_for_kernel(self, kernel_var):
        return self._get_variable_for_kernel(kernel_var, 'gamma')

    def get_pred_loss_and_acc(self, image_batch, label_batch):
        assert False

    def get_pred_loss(self, logits, labels, reduction='weighted_sum_by_nonzero_weights'):
        assert False

    def get_tower_loss_and_acc(self, scope, image_batch, label_batch, tower_name='tower'):
        assert False

    def get_tower_total_loss_and_pred_loss_acc(self, scope, image_batch, label_batch, tower_name='tower'):
        assert False

    def get_value_or_default(self, tensor, default, lens=None):
        v = self.get_value(tensor)
        if v is None:
            assert lens is not None
            return np.ones(lens) * default
        else:
            return v

    def clear(self):
        self.sess.close()
        tf.reset_default_graph()
        print('model cleared')

    def __del__(self):
        self.clear()

    def show_variables(self):
        vs = self.get_global_variables()
        print('-------------------showing global variables---------------')
        for v in vs:
            print(v.name, v.get_shape())
        print('-------------------done showing variables ----------------')

    def _get_dic_of_variables(self, vars):
        result = {}
        values = self.get_value(vars)
        for var, value in zip(vars, values):
            result[var.name] = value
        if self.params.deps is not None:
            print('putting deps={} to the save dict'.format(np.array(self.params.deps)))
            result['deps'] = np.array(self.params.deps)
        return result

    def save_weights_to_hdf5(self, hdf5_file):
        all_key_vars = self.get_key_variables()
        save_vars = []
        for v in all_key_vars:
            if v.name.startswith('v') and not v.name.startswith('v0'):
                continue
            else:
                save_vars.append(v)
        save_dict = self._get_dic_of_variables(save_vars)
        print('got the dict of variable values')
        print('start writing the hdf5 file')
        save_hdf5(save_dict, hdf5_file)
        print('save {} key variables to hdf5 file {}'.format(len(save_dict), hdf5_file))

    def save_weights_and_extra(self, hdf5_file, extra_dict):
        all_key_vars = self.get_key_variables()
        save_vars = []
        for v in all_key_vars:
            if v.name.startswith('v') and not v.name.startswith('v0'):
                continue
            else:
                save_vars.append(v)
        save_dict = self._get_dic_of_variables(save_vars)
        num_key_var = len(save_dict)
        save_dict.update(extra_dict)
        assert len(extra_dict) == len(save_dict) - num_key_var
        print('got the dict of variable values')
        print('start writing the hdf5 file')
        save_hdf5(save_dict, hdf5_file)
        print('save {} key variables and {} others to hdf5 file {}'.format(num_key_var, len(extra_dict), hdf5_file))


    def save_moving_average_weights_to_hdf5(self, hdf5_file, moving_averages):
        key_variables = self.get_key_variables()
        names = []
        fetches = []
        for kv in key_variables:
            if kv.name.startswith('v') and not kv.name.startswith('v0'):
                continue
            names.append(kv.name)
            mav = moving_averages.average(kv)
            if mav is None:
                fetches.append(kv)
            else:
                fetches.append(mav)
        result = self._get_dic_of_variables(fetches)
        save_hdf5(result, hdf5_file)
        print('save {} moving average key variables to hdf5 file {}'.format(len(result), hdf5_file))


    def _init_expma_vars(self, dic, tensors, values, ignore_patterns):
        expma_vars = self.get_expma_vars()
        assert '/ExponentialMovingAverage' not in ignore_patterns
        print('start to init {} ExponentialMovingAverage variables'.format(len(expma_vars)))
        for v in expma_vars:
            name = eliminate_all_patterns_and_starting_vs(v.name, ignore_patterns)
            if name in dic:
                tensors.append(v)
                values.append(dic[name])
            else:
                base_name = name.replace('/ExponentialMovingAverage', '')
                if base_name in dic:
                    print('in the value dict got no original expma value for {} but got the base var {}'.format(v.name, base_name))
                    tensors.append(v)
                    values.append(dic[base_name])
                else:
                    print('no original nor base value for ', v.name)
        print('initializing {} expma vars'.format(len(expma_vars)))


    def _init_from_hdf5_vars_and_values(self, hdf5_file):
        # value_ignore_patterns = ['tower_[0-9]/', ':0', 'cg/', 'conv2d/', re.compile('batchnorm(\d+)/'), '/ExponentialMovingAverage']
        # var_ignore_patterns = ['tower_[0-9]/', ':0', 'cg/', 'conv2d/', re.compile('batchnorm(\d+)/')]
        value_ignore_patterns = ['tower_[0-9]/', ':0', 'cg/', '/ExponentialMovingAverage']
        var_ignore_patterns = ['tower_[0-9]/', ':0', 'cg/']
        self.init_hdf5 = hdf5_file
        vars = self.get_key_variables()
        _dic = read_hdf5(hdf5_file)
        dic = {}
        for k, v in _dic.items():
            dic[eliminate_all_patterns_and_starting_vs(k, value_ignore_patterns)] = v
        tensors = []
        values = []
        for t in vars:
            name = eliminate_all_patterns_and_starting_vs(t.name, var_ignore_patterns)
            if name in dic:
                tensors.append(t)
                values.append(dic[name])
                print('ready to load: ', name, t.get_shape(), dic[name].shape)
                # print(name)
            else:
                print('cannot find matched value for variable ', name)
        print('loaded hdf5, {} matched key vars and {} values'.format(len(tensors), len(values)))

        self._init_expma_vars(dic, tensors, values, var_ignore_patterns)

        return tensors, values


    def get_expma_vars(self):
        expma_vars = [v for v in self.get_global_variables() if '/ExponentialMovingAverage' in v.name]
        return expma_vars


    def init_from_hdf5_op(self, hdf5_file):
        vars, values = self._init_from_hdf5_vars_and_values(hdf5_file)
        ops = []
        for var, value in zip(vars, values):
            # if 'Conv2d_1c_1x1' not in var.name:     #TODO ugly hack
            ops.append(var.assign(value))
        print('prepared the init_from_hdf5_op')
        return tf.group(*ops, name='init_from_hdf5_op')


    def load_weights_from_hdf5(self, hdf5_file):
        vars, values = self._init_from_hdf5_vars_and_values(hdf5_file=hdf5_file)
        if vars:
            self.batch_set_value(vars, values)
        self.last_weights_file = hdf5_file
        print('successfully loaded hdf5. assigned {} variables out of {} total variables'.format(len(vars), len(self.get_global_variables())))

    #   TODO kernel/weights and bias only
    def load_weights_by_order(self, hdf5_file):
        #   keys:   gamma_0, gamma_1, kernel_0, kernel_1 ....
        hdf5_dict = read_hdf5(hdf5_file)
        k_namelist = []
        b_namelist = []
        for name, value in hdf5_dict.items():
            if 'kernel' in name:
                k_namelist.append(name)
            elif 'bias' in name:
                b_namelist.append(name)
            # else:
            #     assert False
        k_namelist.sort(key=lambda x: int(x.split('_')[1]))
        b_namelist.sort(key=lambda x: int(x.split('_')[1]))

        k_vs = []
        k_assign = []
        b_vs = []
        b_assign = []

        vars = self.get_key_variables()
        for v in vars:
            if 'kernel' in v.name or 'weights' in v.name:
                k_vs.append(v)
                k_assign.append(hdf5_dict[k_namelist.pop(0)])
            elif 'bias' in v.name:
                b_vs.append(v)
                b_assign.append(hdf5_dict[b_namelist.pop(0)])

        init_vars = k_vs + b_vs
        init_values = k_assign + b_assign

        #   other vars
        value_ignore_patterns = ['tower_[0-9]/', ':0', 'cg/', '/ExponentialMovingAverage', re.compile('batchnorm(\d+)/')]
        var_ignore_patterns = ['tower_[0-9]/', ':0', 'cg/', re.compile('batchnorm(\d+)/')]
        self.init_hdf5 = hdf5_file
        vars = self.get_key_variables()
        other_dic = {}
        for k, v in hdf5_dict.items():
            other_dic[eliminate_all_patterns_and_starting_vs(k, value_ignore_patterns)] = v
        for t in vars:
            name = eliminate_all_patterns_and_starting_vs(t.name, var_ignore_patterns)
            if name in other_dic:
                init_vars.append(t)
                init_values.append(other_dic[name])
        self.batch_set_value(init_vars, init_values)
        print('loaded {} weights by order'.format(len(init_vars)))
        self.last_weights_file = hdf5_file







    def set_value(self, t, value):
        with self.graph.as_default():
            ph = tf.placeholder(t.dtype, t.get_shape())
            op = t.assign(ph)
            self.sess.run(op, feed_dict={ph: value})

    def batch_set_value(self, tensors, values):
        print('got {} vars and {} values to assign'.format(len(tensors), len(values)))
        assert len(tensors) == len(values)
        with self.graph.as_default():
            ops = []
            feed_dict = {}
            for t, v in zip(tensors, values):
                if t is not None:
                    ph = tf.placeholder(t.dtype, t.get_shape())
                    ops.append(t.assign(ph))
                    feed_dict[ph] = v
            self.sess.run(ops, feed_dict=feed_dict)

    def get_value(self, t):
        if t is None:
            return None
        if not (type(t) is list or type(t) is tuple):
            return self.sess.run(t)
        fetch_list = []
        none_idxes = []
        for i, itm in enumerate(t):
            if itm is None:
                none_idxes.append(i)
            else:
                fetch_list.append(itm)
        print('fetch_list len: ', len(fetch_list))
        values = self.sess.run(fetch_list)
        print('fetched len: ', len(values))
        result = []
        values_i = 0
        for i in range(len(t)):
            if i in none_idxes:
                result.append(None)
            else:
                result.append(values[values_i])
                values_i += 1
        return result

    def get_key_variables(self):
        result = self.get_trainable_variables()
        result += self.get_moving_mean_variables()
        result += self.get_moving_variance_variables()
        return result

    # def get_post_init_ops(self):
    #     # Copy initialized values for variables on GPU 0 to other GPUs.
    #     global_vars = tf.global_variables()
    #     var_by_name = dict([(v.name, v) for v in global_vars])
    #     post_init_ops = []
    #     for v in global_vars:
    #         split_name = v.name.split('/')
    #         # TODO(b/62630508): use more specific prefix than v or v0.
    #         if split_name[0] == 'v0' or not v.name.startswith('v'):
    #             continue
    #         split_name[0] = 'v0'
    #         copy_from = var_by_name['/'.join(split_name)]
    #         post_init_ops.append(v.assign(copy_from.read_value()))
    #     post_init_ops += self._warmup_ops
    #     return post_init_ops



    def run(self):
        """Run the benchmark task assigned to this process.

        Returns:
          Dictionary of statistics for training or eval.
        Raises:
           ValueError: unrecognized job name.
        """
        raise NotImplementedError('Must be implemented in derived classes')



    def eval_loop(self, input_producer_op, enqueue_ops, fetches):
        """Evaluate a model every self.params.eval_interval_secs.

        Returns:
          Dictionary containing eval statistics. Currently returns an empty
          dictionary.
        """
        raise NotImplementedError('Must be implemented in derived classes')



    def eval_once(self, saver, summary_writer, target, local_var_init_op_group,
                  input_producer_op, enqueue_ops, fetches, summary_op, eval_feed_dict=None):
        """Evaluate the model from a checkpoint using validation dataset."""
        raise NotImplementedError('Must be implemented in derived classes')


    GPU_CACHED_INPUT_VARIABLE_NAME = 'gpu_cached_inputs'



    def do_train(self, graph_info):
        """Benchmark the graph.

        Args:
          graph_info: the namedtuple returned by _build_graph() which
            contains all necessary information to benchmark the graph, including
            named tensors/ops list, fetches, etc.
        Returns:
          Dictionary containing training statistics (num_workers, num_steps,
          average_wall_time, images_per_sec).
        """
        raise NotImplementedError('Must be implemented in derived classes')



    def add_forward_pass_and_gradients(self,
                                       phase_train,
                                       rel_device_num,
                                       abs_device_num,
                                       input_processing_info,
                                       gpu_compute_stage_ops,
                                       gpu_grad_stage_ops):
        raise NotImplementedError('Must be implemented in derived classes')



    def get_input_preprocessor(self):
        raise NotImplementedError('Must be implemented in derived classes')







    #   shawn
    #   note:   if an op should be regarded as some kind of accuracy, its name should start with 'accuracy:'    (original)
    #           if you wish to print training accuracy, set params.print_training_accuracy = True
    #           if an op should be regarded as some kind of loss, its name should start with 'loss_term:'    (by shawn)
    #           other outputs which we wish to fetch should start with 'other:'.
    #           all_loss_terms and all_others will be fetched from the first device only (not averaged across all devices), and regardless of phase_train
    def _build_model(self):
        """Build the TensorFlow graph."""
        if self.datasets_use_prefetch:
            assert not self.params.staged_vars
            assert not self.variable_mgr.supports_staged_vars()

        # Adjust seed so different workers start read different input files.
        if self.params.variable_update == 'horovod':
            import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
            seed_adjustment = hvd.rank()
        else:
            seed_adjustment = 0
        tf.set_random_seed(self.params.tf_random_seed + seed_adjustment)
        np.random.seed(4321 + seed_adjustment)
        phase_train = not (self.params.eval or self.params.forward_only)

        log_fn('Generating model')
        losses = []
        device_grads = []
        all_logits = []
        all_accuracy_ops = {}
        gpu_compute_stage_ops = []
        gpu_grad_stage_ops = []

        #   shawn, maybe useful oneday
        #   all_loss_terms should be
        all_others = {}
        all_loss_terms = {}

        with tf.device(self.global_step_device):
            global_step = tf.train.get_or_create_global_step()
            self._maybe_initialize_fp16()

        # Build the processing and model for the worker.
        input_producer_op = None
        with tf.name_scope('input_processing'):
            input_processing_info = self._build_input_processing(shift_ratio=0)
            if input_processing_info.input_producer_op is not None:
                input_producer_op = tf.group(*input_processing_info.input_producer_op)
            # summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
            # print('the summary op after _build_input_processing:', summary_ops)


        update_ops = None
        staging_delta_ops = []

        for device_num in range(len(self.devices)):
            with tf.name_scope('tower_%i' % device_num) as name_scope, (
                    self.variable_mgr.create_outer_variable_scope(device_num)):
                results = self.add_forward_pass_and_gradients(
                    phase_train, device_num, device_num, input_processing_info,
                    gpu_compute_stage_ops, gpu_grad_stage_ops)

                if self.params.backbone_model_path:
                    self.model.add_backbone_saver()

                losses.append(results['loss'])
                if phase_train:
                    device_grads.append(results['gradvars'])
                else:
                    print('****************the logits shape is ', results['logits'].get_shape())
                    all_logits.append(results['logits'])

                #   shawn
                if device_num == 0:
                    for _op_name, _op in results.items():
                        if 'loss_term:' in _op_name:
                            all_loss_terms[_op_name] = _op
                        elif 'other:' in _op_name:
                            all_others[_op_name] = _op

                if not phase_train or self.params.print_training_accuracy:
                    for name, op in results.items():
                        if name.startswith('accuracy:'):
                            key = name[9:]
                            if key not in all_accuracy_ops:
                                all_accuracy_ops[key] = []
                            all_accuracy_ops[key].append(op)

                if device_num == 0:
                    # Retain the Batch Normalization updates operations only from the
                    # first tower. These operations update the moving mean and moving
                    # variance variables, which are updated (but not used) during
                    # training, and used during evaluation. The moving mean and variance
                    # approximate the true mean and variance across all images in the
                    # dataset. Therefore, in replicated mode, these moving averages would
                    # be almost identical for each tower, and so we only update and save
                    # the moving averages for one tower. In parameter server mode, all
                    # towers share a copy of the variables so we also only need to update
                    # and save the moving averages once.
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
                    if self.datasets_use_prefetch:
                        assert not self.variable_mgr.staging_delta_ops
                    else:
                        staging_delta_ops = list(self.variable_mgr.staging_delta_ops)

        enqueue_ops = []
        if not self.datasets_use_prefetch:
            if self.variable_mgr.supports_staged_vars():
                for staging_ops in self.variable_mgr.staging_vars_on_devices:
                    gpu_compute_stage_ops.extend(
                        [put_op for _, (put_op, _) in six.iteritems(staging_ops)])
            enqueue_ops.append(tf.group(*gpu_compute_stage_ops,
                name='gpu_compute_stage_ops_group'))
            if gpu_grad_stage_ops:
                staging_delta_ops += gpu_grad_stage_ops
            if staging_delta_ops:
                enqueue_ops.append(tf.group(*(staging_delta_ops)))

        fetches = self._build_fetches(global_step, all_logits, losses, device_grads,
            enqueue_ops, update_ops, all_accuracy_ops,
            phase_train, all_loss_terms=all_loss_terms, all_others=all_others)
        if 'internal_outputs' in results:
            assert len(self.devices) == 1
            fetches['internal_outputs'] = results['internal_outputs']

        fetches['labels'] = results['labels']

        print('displaying fetches: {}'.format(fetches))

        self.update_name_to_variables()
        return (input_producer_op, enqueue_ops, fetches)





    #   shawn
    #   the gradient-averaging step takes place here
    #   all accuracies will be averaged by self.batch_size and summarized
    #   but all_others and all_loss_terms will not be averaged
    #   self.gradient_handler is expected to handl gradients properly on the specified device (since it is within the device scope)
    def _build_fetches(self, global_step, all_logits, losses, device_grads,
                       enqueue_ops, update_ops, all_accuracy_ops, phase_train,
                       all_loss_terms=None, all_others=None):

        """Complete construction of model graph, populating the fetches map."""
        fetches = {}

        def maybe_update_all():
            if all_loss_terms is not None:
                fetches.update(all_loss_terms)
            if all_others is not None:
                fetches.update(all_others)

        if enqueue_ops:
            fetches['enqueue_ops'] = enqueue_ops
        for name, ops in all_accuracy_ops.items():
            # For fetches that starts with 'tensor:', keep dimension and skip reducing
            # them to scalars.
            if name.startswith(constants.UNREDUCED_ACCURACY_OP_PREFIX):
                fetches[name[len(constants.UNREDUCED_ACCURACY_OP_PREFIX):]] = ops[0]
            else:
                #   shawn TODO
                # fetches[name] = ops
                fetches[name] = tf.reduce_sum(ops) / self.batch_size
                if self.task_index == 0 and self.params.summary_verbosity >= 1:
                    log_fn('write summary for accuracy {}'.format(name))
                    tf.summary.scalar(name, fetches[name])

        if not phase_train:
            if self.params.forward_only:
                fetches['all_logits'] = tf.concat(all_logits, 0)
                maybe_update_all()
            fetches['loss'] = tf.reduce_mean(losses)
            fetches['logits'] = tf.concat(all_logits, 0)    # TODO shawn added this line 20190313
            return fetches

        apply_gradient_devices, gradient_state = (
            self.variable_mgr.preprocess_device_grads(device_grads))

        training_ops = []
        self.update_name_to_variables()
        for d, device in enumerate(apply_gradient_devices):
            with tf.device(device):
                with tf.name_scope('average_loss'):
                    average_loss = tf.reduce_mean(losses)
                with tf.name_scope('get_gradients_to_apply'):
                    avg_grads = self.variable_mgr.get_gradients_to_apply(d,
                        gradient_state)

                gradient_clip = self.params.gradient_clip
                # TODO(reedwm): Greatly simplify the learning rate code.
                if (self.params.variable_update == 'horovod' or
                            self.params.variable_update == 'collective_all_reduce'):
                    # Each worker independently increments global_step.
                    examples_per_step = self.batch_size * self.num_workers
                else:
                    # global_step is shared by all workers, and so every iteration
                    # global_step is incremented by num_workers.
                    examples_per_step = self.batch_size
                learning_rate, boundaries = get_learning_rate(self.params, global_step,
                    self.dataset.num_examples_per_epoch(),
                    self.model, examples_per_step)

                self.lr_boundaries = boundaries

                if gradient_clip is not None:
                    with tf.name_scope('clip_gradients'):
                        clipped_grads = [(tf.clip_by_value(grad, -gradient_clip,
                            +gradient_clip), var)
                                         for grad, var in avg_grads]
                else:
                    clipped_grads = avg_grads

                #   2X some gradients
                clipped_grads = double_keyword_gradients(clipped_grads, self.my_params.double_gradient_keywords)


                #   by shawn
                if self.gradient_handler is not None:
                    clipped_grads = self.gradient_handler.handle_gradient(clipped_grads, d, device)
                if d == 0:
                    maybe_update_all()

                learning_rate = tf.identity(learning_rate, name='learning_rate_tensor')
                opt = get_optimizer(self.params, learning_rate)
                loss_scale_params = variable_mgr_util.AutoLossScaleParams(
                    enable_auto_loss_scale=self.enable_auto_loss_scale,
                    loss_scale=self.loss_scale,
                    loss_scale_normal_steps=self.loss_scale_normal_steps,
                    inc_loss_scale_every_n=self.params.fp16_inc_loss_scale_every_n,
                    is_chief=not self.job_name or self.task_index == 0)

                with tf.name_scope('append_apply_gradient_ops'):
                    self.variable_mgr.append_apply_gradients_ops(
                        gradient_state, opt, clipped_grads, training_ops,
                        loss_scale_params)

        train_op = tf.group(*(training_ops + update_ops), name='train_ops_group')

        with tf.device(self.cpu_device):
            if self.task_index == 0 and self.params.summary_verbosity >= 1:
                tf.summary.scalar('learning_rate', learning_rate)
                tf.summary.scalar(self.params.loss_type_to_report, average_loss)

                self.add_other_summaries()

                if self.loss_scale is not None:
                    tf.summary.scalar('loss_scale', self.loss_scale)
                if self.loss_scale_normal_steps:
                    tf.summary.scalar('loss_scale_normal_steps',
                        self.loss_scale_normal_steps)

                if self.params.summary_verbosity >= 2:
                    for var in tf.trainable_variables():
                        tf.summary.histogram(var.op.name, var)


                if self.params.summary_verbosity >= 3:
                    self.gradient_histogram_summary(avg_grads)
                    for grad, var in avg_grads:
                        if grad is not None:
                            tf.summary.histogram(var.op.name + '/gradients', grad)


        fetches['train_op'] = train_op
        fetches['average_loss'] = average_loss

        log_fn('-------------start displaying the fetch dict----------------')
        for name, ops in fetches.items():
            log_fn('the fetched name is {}'.format(name))
        log_fn('-------------done displaying the fetch dict----------------')

        return fetches


    def add_other_summaries(self):
        print('no other summaries')


    #   ------------------------- the following member functions should never never never be touched

    def _build_input_processing(self, shift_ratio=0):
        """"Build the image (pre)processing portion of the model graph.

        Args:
          shift_ratio: shift_ratio for data_flow_ops.RecordInput.

        Returns:
          An InputProcessingInfo containing all the input sources to the model.
        """
        input_processing_info = InputProcessingInfo(
            input_producer_op=None,
            input_producer_stages=None,
            function_buffering_resources=None,
            multi_device_iterator_input=None)

        # If using synthetic gpu inputs, do nothing on the cpu side.
        if self.dataset.use_synthetic_gpu_inputs():
            assert not self.datasets_use_prefetch
            return input_processing_info

        # Use prefetching mechanism provided by dataset input pipeline.
        if self.datasets_use_prefetch:
            if self.params.use_multi_device_iterator:
                multi_device_iterator = (
                    self.input_preprocessor.build_multi_device_iterator(
                        self.batch_size, len(self.devices), self.cpu_device,
                        self.params, self.raw_devices, self.dataset, subset=self.subset))
                return input_processing_info._replace(
                    multi_device_iterator_input=multi_device_iterator.get_next())

            function_buffering_resources = (
                self.input_preprocessor.build_prefetch_input_processing(
                    self.batch_size, self.model.get_input_shapes(),
                    len(self.devices), self.cpu_device, self.params, self.devices,
                    self.model.get_input_data_types(), self.dataset))
            return input_processing_info._replace(
                function_buffering_resources=function_buffering_resources)

        # Not using dataset prefetching. Use a staging area to mimic the prefetching
        # behavior instead.
        with tf.device(self.cpu_device):
            input_list = self.input_preprocessor.minibatch(
                self.dataset,
                subset=self.subset,
                # TODO(laigd): consider removing this option, it should always use
                # datasets.
                use_datasets=self.params.use_datasets,
                datasets_repeat_cached_sample=(
                    self.params.datasets_repeat_cached_sample),
                shift_ratio=shift_ratio)

            input_producer_op = []
            input_producer_stages = []
            for device_num in range(len(self.devices)):
                staging_area = data_flow_ops.StagingArea(
                    [parts[0].dtype for parts in input_list],
                    shapes=[parts[0].get_shape() for parts in input_list],
                    shared_name='input_producer_staging_area_%d' % device_num)
                input_producer_stages.append(staging_area)
                for group_index in xrange(self.batch_group_size):
                    batch_index = group_index + device_num * self.batch_group_size
                    put_op = staging_area.put(
                        [parts[batch_index] for parts in input_list])
                    input_producer_op.append(put_op)
            assert input_producer_op

        return input_processing_info._replace(
            input_producer_op=input_producer_op,
            input_producer_stages=input_producer_stages)


    # TODO(laigd): this changes the global device list which is used everywhere,
    # consider refactoring it.
    def reset_devices_for_task(self, task_num, is_local=False):
        """Used to imitate another task when building a distributed graph."""
        worker_prefix = ('/job:localhost' if is_local else
                         '/job:worker/replica:0/task:%s' % task_num)
        self.cpu_device = '%s/cpu:0' % worker_prefix
        self.raw_devices = [
            '%s/%s:%i' % (worker_prefix, self.params.device, i)
            for i in xrange(self.num_gpus)
        ]
        self.devices = self.variable_mgr.get_devices()


    def raw_devices_across_tasks(self, is_local=False):
        """Returns list of raw device names across all tasks."""
        if is_local:
            assert self.num_workers == 1
            return self.raw_devices
        else:
            return [
                'job:worker/replica:0/task%s/%s:%i' % (t, self.params.device, i)
                for t in xrange(self.num_workers)
                for i in xrange(self.num_gpus)
            ]

    def print_info(self):
        """Print basic information."""
        benchmark_info = self._get_params_info()
        log_fn('Model:       %s' % self.model.get_model_name())
        log_fn('Dataset:     %s' % benchmark_info['dataset_name'])
        log_fn('Mode:        %s' % get_mode_from_params(self.params))
        log_fn('SingleSess:  %s' % benchmark_info['single_session'])
        log_fn('Batch size:  %s global' % (self.batch_size * self.num_workers))
        log_fn('             %s per device' % (self.batch_size /
                                               len(self.raw_devices)))
        if self.batch_group_size > 1:
            log_fn('             %d batches per prepocessing group' %
                   self.batch_group_size)
        log_fn('Num batches: %d' % self.num_batches)
        log_fn('Num epochs:  %.2f' % self.num_epochs)
        log_fn('Devices:     %s' % benchmark_info['device_list'])
        log_fn('Data format: %s' % self.params.data_format)
        if self.rewriter_config:
            log_fn('RewriterConfig: %s' % self.rewriter_config)
        log_fn('Optimizer:   %s' % self.params.optimizer)
        log_fn('Variables:   %s' % self.params.variable_update)
        if (self.params.variable_update == 'replicated' or
                    self.params.variable_update == 'distributed_all_reduce'
            or self.params.variable_update == 'collective_all_reduce'):
            log_fn('AllReduce:   %s' % self.params.all_reduce_spec)
        if self.job_name:
            log_fn('Sync:        %s' % self.params.cross_replica_sync)
        if self.params.staged_vars:
            log_fn('Staged vars: %s' % self.params.staged_vars)
        if self.params.variable_update == 'horovod' and self.params.horovod_device:
            log_fn('Horovod on:  %s' % self.params.horovod_device)
        log_fn('==========')


    def gradient_histogram_summary(self, avg_grads):
        """Create histogram of log values of all non-zero gradients."""
        with tf.name_scope('log_gradients_summary'):
            all_grads = []
            for grad, _ in avg_grads:
                all_grads.append(tf.reshape(grad, [-1]))
            grads = tf.abs(tf.concat(all_grads, 0))
            # exclude grads with zero values.
            indices_for_non_zero_grads = tf.where(tf.not_equal(grads, 0))
            log_grads = tf.reshape(
                tf.log(tf.gather(grads, indices_for_non_zero_grads)), [-1])
            tf.summary.histogram('log_gradients', log_grads)


    def add_sync_queues_and_barrier(self, name_prefix, enqueue_after_list):
        """Adds ops to enqueue on all worker queues.

        Args:
          name_prefix: prefixed for the shared_name of ops.
          enqueue_after_list: control dependency from ops.

        Returns:
          An op that should be used as control dependency before starting next step.
        """
        self.sync_queue_counter += 1
        with tf.device(self.sync_queue_devices[(
                    self.sync_queue_counter % len(self.sync_queue_devices))]):
            sync_queues = [
                tf.FIFOQueue(self.num_workers, [tf.bool], shapes=[[]],
                    shared_name='%s%s' % (name_prefix, i))
                for i in range(self.num_workers)]
            queue_ops = []
            # For each other worker, add an entry in a queue, signaling that it can
            # finish this step.
            token = tf.constant(False)
            with tf.control_dependencies(enqueue_after_list):
                for i, q in enumerate(sync_queues):
                    if i == self.task_index:
                        queue_ops.append(tf.no_op())
                    else:
                        queue_ops.append(q.enqueue(token))

            # Drain tokens off queue for this worker, one for each other worker.
            queue_ops.append(
                sync_queues[self.task_index].dequeue_many(len(sync_queues) - 1))

            return tf.group(*queue_ops)


    #   shawn
    #   temporarily this function should never be called, because I do not know what 'distributed_all_reduce' is
    def _build_model_single_session(self):
        """Build the TensorFlow graph for multiple replicas in a single_session.

        Returns:
          input_producer_op:
          enqueue_ops:
          fetches:

        Raises:
           ValueError: optimizer not recognized.

        Single session runs multiple model replicas as part of one large
        distributed graph, whose global execution is always step-synchronized.
        """
        # verify assumptions
        assert self.params.task_index == 0
        assert not self.params.eval
        assert not self.params.forward_only
        assert not self.params.staged_vars

        tf.set_random_seed(self.params.tf_random_seed)
        np.random.seed(4321)
        phase_train = True

        log_fn('Generating model')
        losses = []
        device_grads = []
        all_logits = []
        all_accuracy_ops = {}
        gpu_compute_stage_ops = []
        gpu_grad_stage_ops = []

        with tf.device(self.global_step_device):
            global_step = tf.train.get_or_create_global_step()

        update_ops = []
        global_input_producer_op = []

        is_local = not self.job_name
        if is_local:
            assert self.num_workers == 1
        for task_num in range(self.num_workers):
            # Reset the devices that self.variable_mgr knows about to those
            # belonging to the next worker (task).
            self.reset_devices_for_task(task_num, is_local)
            # Build the per-worker image processing
            with tf.name_scope('input_processing'):
                input_processing_info = self._build_input_processing(
                    shift_ratio=(float(task_num) / self.num_workers))
            if input_processing_info.input_producer_op is not None:
                global_input_producer_op.extend(input_processing_info.input_producer_op)
            # Build the per-worker model replica.
            for rel_device_num in range(len(self.devices)):
                abs_device_num = task_num * len(self.devices) + rel_device_num
                with self.variable_mgr.create_outer_variable_scope(
                        abs_device_num), tf.name_scope(
                            'task_%i_tower_%i' % (task_num, rel_device_num)) as name_scope:
                    task_results = self.add_forward_pass_and_gradients(
                        phase_train, rel_device_num, abs_device_num,
                        input_processing_info, gpu_compute_stage_ops, gpu_grad_stage_ops)

                    if self.params.backbone_model_path:
                        self.model.add_backbone_saver()

                    if phase_train:
                        losses.append(task_results['loss'])
                        device_grads.append(task_results['gradvars'])
                    else:
                        all_logits.append(task_results['logits'])
                    if not phase_train or self.params.print_training_accuracy:
                        for name, op in task_results.items():
                            if name.startswith('accuracy:'):
                                key = name[9:]
                                if key not in all_accuracy_ops:
                                    all_accuracy_ops[key] = []
                                all_accuracy_ops[key].append(op)

                    if rel_device_num == 0:
                        # Retain the Batch Normalization updates operations only
                        # from the first tower. These operations update the moving
                        # mean and moving variance variables, which are updated
                        # (but not used) during training, and used during
                        # evaluation. The moving mean and variance approximate the
                        # true mean and variance across all images in the
                        # dataset. Therefore, in replicated mode, these moving
                        # averages would be almost identical for each tower, and
                        # so we only update and save the moving averages for one
                        # tower. In parameter server mode, all towers share a copy
                        # of the variables so we also only need to update and save
                        # the moving averages once.
                        update_ops.extend(
                            tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope))
                        assert not self.variable_mgr.staging_delta_ops

        enqueue_ops = []
        if gpu_compute_stage_ops:
            enqueue_ops.append(tf.group(*gpu_compute_stage_ops,
                name='gpu_compute_stage_ops'))
        assert not self.variable_mgr.supports_staged_vars()
        assert not gpu_grad_stage_ops

        fetches = self._build_fetches(global_step, all_logits, losses, device_grads,
            enqueue_ops, update_ops, all_accuracy_ops,
            phase_train)
        if global_input_producer_op:
            global_input_producer_op = tf.group(*global_input_producer_op)
        else:
            global_input_producer_op = None
        return (global_input_producer_op, enqueue_ops, fetches)


    def _maybe_initialize_fp16(self):
        """Initialize fp16 settings."""
        if self.params.use_fp16:
            init_loss_scale_val = float(self.params.fp16_loss_scale or
                                        self.model.get_fp16_loss_scale())
            self.loss_scale = None
            self.loss_scale_normal_steps = None
            if self.enable_auto_loss_scale or init_loss_scale_val != 1:
                self.loss_scale = tf.get_variable(
                    name='loss_scale',
                    initializer=init_loss_scale_val,
                    dtype=tf.float32,
                    trainable=False)
            if self.enable_auto_loss_scale:
                self.loss_scale_normal_steps = tf.get_variable(
                    name='loss_scale_normal_steps', initializer=0, trainable=False)


    #   shawn
    def _preprocess_graph(self, graph, graph_info):
        """Preprocess the graph before executing.

        Depending on the params, it runs various preprocessing on the graph,
        including freezing, TensorRT conversion, etc.

        Args:
          graph: the graph to preprocess.
          graph_info: the namedtuple returned by _build_graph() which
            contains all necessary information to benchmark the graph, including
            named tensors/ops list, fetches, etc.

        Returns:
          The updated graph and graph_info with the ops/tensors/fetches updated
          according to the imported graph.
        """
        assert isinstance(graph_info.fetches, dict)
        assert isinstance(graph_info.global_step, tf.Variable)
        if not self.forward_only_and_freeze:    #TODO note this !
            return (graph, graph_info)

        # Get the names of the ops that need to keep during conversion.
        flattened_op_names = list(
            set([
                v.name.split(':')[0]
                for v in nest.flatten(graph_info)
                if v is not None
            ]))
        # Get variables that we don't want to freeze.
        # Only keep unfreezable variables in forward_only_and_freeze mode.
        # TODO(laigd): consider making global_step a constant.
        variables_to_keep = {graph_info.global_step: tf.GraphKeys.GLOBAL_VARIABLES}
        variables_to_keep.update({
            local_variable: tf.GraphKeys.LOCAL_VARIABLES
            for local_variable in self._unfreezable_local_variables(graph)
        })

        variable_initializers = [
            variable.initializer.name for variable in variables_to_keep]
        output_node_names = (
            flattened_op_names +
            # Add variable initializer and read ops to the output list, so
            # convert_variables_to_constants() will keep them.
            variable_initializers +
            [variable.value().op.name for variable in variables_to_keep])
        graphdef = graph.as_graph_def(add_shapes=True)

        # Freeze the graph.
        with graph.as_default():
            with tf.Session(config=create_config_proto(self.params)) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                graphdef = graph_util.convert_variables_to_constants(
                    sess,
                    graphdef,
                    output_node_names,
                    variable_names_blacklist=[
                        variable.op.name for variable in variables_to_keep
                    ])

        # Run TensorRT conversion.
        if self.params.trt_mode:
            print('using trt_mode')
            # Import here instead of at top, because this will crash if TensorRT is
            # not installed
            from tensorflow.contrib import tensorrt as trt  # pylint: disable=g-import-not-at-top
            # Avoid TF-TRT bridge from touching all variable initializer ops and their
            # dependencies, since they can directly be fetched by sess.run()s that
            # initialize the variables.
            # pylint: disable=protected-access
            name_to_input_name, _, _ = graph_util_impl._extract_graph_summary(
                graphdef)
            initializer_subgraph_ops = graph_util_impl._bfs_for_reachable_nodes(
                variable_initializers, name_to_input_name)
            # pylint: enable=protected-access

            graphdef = trt.create_inference_graph(
                graphdef,
                outputs=output_node_names + list(initializer_subgraph_ops),
                max_batch_size=self.model.get_batch_size(),
                max_workspace_size_bytes=self.params.trt_max_workspace_size_bytes,
                precision_mode=self.params.trt_mode)
        else:
            print('no trt_mode')

        # Creates a new graph as the default and import the converted graph back.
        print('new a graph')
        updated_graph = tf.Graph()

        def _get_tensors_or_ops(inputs):
            """Gets the updated tensors or ops from 'updated_graph'."""

            def _get_fn(element):
                if element is None:
                    return None
                if ':' in element.name:
                    return updated_graph.get_tensor_by_name(element.name)
                return updated_graph.get_operation_by_name(element.name)

            if isinstance(inputs, (list, dict, tuple)):
                return nest.map_structure(_get_fn, inputs)
            else:
                return _get_fn(inputs)

        with updated_graph.as_default():
            importer.import_graph_def(graph_def=graphdef, name='')

            # Update the variables
            for variable in variables_to_keep:
                updated_variable = tf.Variable.from_proto(variable.to_proto())
                tf.add_to_collection(variables_to_keep[variable], updated_variable)
                if variable is graph_info.global_step:
                    updated_global_step = updated_variable



        updated_graph_info = GraphInfo(
            input_producer_op=_get_tensors_or_ops(graph_info.input_producer_op),
            enqueue_ops=_get_tensors_or_ops(graph_info.enqueue_ops),
            execution_barrier=_get_tensors_or_ops(graph_info.execution_barrier),
            local_var_init_op_group=_get_tensors_or_ops(
                graph_info.local_var_init_op_group),
            fetches=_get_tensors_or_ops(graph_info.fetches),
            global_step=updated_global_step,
            mvav_op=_get_tensors_or_ops(graph_info.mvav_op))
        return (updated_graph, updated_graph_info)


    def _config_benchmark_logger(self):
        """Config the model garden benchmark logger."""
        model_benchmark_logger = None
        if self.params.benchmark_log_dir is not None:
            try:
                from official.utils.logs import logger as models_logger  # pylint: disable=g-import-not-at-top
            except ImportError:
                tf.logging.fatal('Please include tensorflow/models to the PYTHONPATH '
                                 'in order to use BenchmarkLogger. Configured '
                                 'benchmark_log_dir: %s'
                                 % self.params.benchmark_log_dir)
                raise
            model_benchmark_logger = models_logger.BenchmarkFileLogger(
                self.params.benchmark_log_dir)
        self.benchmark_logger = model_benchmark_logger


    def _get_params_info(self):
        """Get the common parameters info for the benchmark run.

        Returns:
          A dict of processed parameters.
        """
        dataset_name = self.dataset.name
        if self.dataset.use_synthetic_gpu_inputs():
            dataset_name += ' (synthetic)'
        single_session = self.params.variable_update == 'distributed_all_reduce'
        if single_session:
            device_list = self.raw_devices_across_tasks()
        elif self.params.variable_update == 'horovod':
            device_list = ['horovod/%s:%d' % (self.params.device, idx)
                           for idx in range(self.num_workers)]
        else:
            device_list = self.raw_devices
        return {
            'dataset_name': dataset_name,
            'single_session': single_session,
            'device_list': device_list, }

    def _log_benchmark_run(self):
        """Log the benchmark info to the logger.

        The info logged here should be similar to print_info(), but in a structured
        JSON format.
        """
        if self.benchmark_logger:
            benchmark_info = self._get_params_info()

            run_param = {
                'model': self.model.get_model_name(),
                'dataset': benchmark_info['dataset_name'],
                'mode': get_mode_from_params(self.params),
                'single_sess': benchmark_info['single_session'],
                'devices': benchmark_info['device_list'],
                'batch_size': self.batch_size,
                'batch_size_per_device': self.batch_size / len(self.raw_devices),
                'num_batches': self.num_batches,
                'num_epochs': self.num_epochs,
                'data_format': self.params.data_format,
                'rewrite_config': self.rewriter_config,
                'optimizer': self.params.optimizer,
                'session_config': create_config_proto(self.params),
            }
            # TODO(scottzhu): tf_cnn_benchmark might execute several times with
            # different param setting on the same box. This will cause the run file to
            # only contain the latest info. The benchmark_log_dir should be updated
            # for every new run.
            self.benchmark_logger.log_run_info(
                self.model.get_model_name(), benchmark_info['dataset_name'],
                run_param, test_id=self.params.benchmark_test_id)



    def _unfreezable_local_variables(self, graph):
        """Get the local variables that we don't want to freeze."""
        return graph.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES,
            # We don't freeze the gpu_cached_images local variable so it won't get
            # constant folded with ops which process the input.
            scope='.*' + BenchmarkBase.GPU_CACHED_INPUT_VARIABLE_NAME)




    def _build_graph(self):
        """Build the graph.

        Returns:
          A namedtuple containing the ops/tensors that required by
          _benchmark_graph().
        """
        if self.params.variable_update == 'distributed_all_reduce':
            self.single_session = True
            (input_producer_op, enqueue_ops, fetches) = (
                self._build_model_single_session())
        else:
            self.single_session = False
            (input_producer_op, enqueue_ops, fetches) = self._build_model()
        fetches_list = nest.flatten(list(fetches.values()))
        main_fetch_group = tf.group(*fetches_list, name='main_fetch_group')
        execution_barrier = None
        if (not self.single_session and self.job_name and
                not self.params.cross_replica_sync):
            execution_barrier = self.add_sync_queues_and_barrier(
                'execution_barrier_', [])

        global_step = tf.train.get_global_step()
        with tf.device(self.global_step_device), tf.name_scope('inc_global_step'):
            with tf.control_dependencies([main_fetch_group]):
                fetches['inc_global_step'] = global_step.assign_add(1)

        if ((not self.single_session) and (not self.distributed_collective) and
                self.job_name and self.params.cross_replica_sync):
            # Block all replicas until all replicas are ready for next step.
            fetches['sync_queues'] = self.add_sync_queues_and_barrier(
                'sync_queues_step_end_', [main_fetch_group])


        if self.my_params.save_mvav:
            print('prepare self.variable_averages')
            self.variable_averages = tf.train.ExponentialMovingAverage(
                0.999, global_step)
            vars_to_average = [v for v in self.get_trainable_variables() if (not v.name.startswith('v')) or v.name.startswith('v0')]
            print('prepare to average these vars {}'.format([v.name for v in vars_to_average]))
            variables_averages_op = self.variable_averages.apply(vars_to_average)
        else:
            print('no need for variable_averages')
            self.variable_averages = None
            variables_averages_op = None

        # Skips the init ops for freezable local variables in forward_only mode so
        # we can remove all the assign ops when converting variables to constants.
        with tf.name_scope('local_variable_initialization'):

            if self.forward_only_and_freeze:
                local_var_init_op = tf.variables_initializer(
                    self._unfreezable_local_variables(tf.get_default_graph()))
            else:
                local_var_init_op = tf.local_variables_initializer()
        table_init_ops = tf.tables_initializer()

        variable_manager_init_ops = [local_var_init_op]


        hdf5_init_op = None
        if self.my_params.init_hdf5:
            if self.my_params.load_ckpt is None:
                print('prepare the op for init from hdf5: ', self.my_params.init_hdf5)
                with tf.control_dependencies([local_var_init_op]):
                    hdf5_init_op = self.init_from_hdf5_op(hdf5_file=self.my_params.init_hdf5)
                    variable_manager_init_ops.append(hdf5_init_op)
            else:
                print('though got the param init_hdf5, but also got load_ckpt, so do not prepare the op for init from hdf5')




        if table_init_ops:
            variable_manager_init_ops.extend([table_init_ops])

        if not self.forward_only_and_freeze:
            dependencies = [hdf5_init_op] if hdf5_init_op is not None else [local_var_init_op]
            if self.my_params.init_global_step:
                print('initialize the global step to ', self.my_params.init_global_step)
                init_global_step_op = global_step.assign(self.my_params.init_global_step)
                variable_manager_init_ops.append(init_global_step_op)
                dependencies.append(init_global_step_op)

            with tf.control_dependencies(dependencies):
                variable_manager_init_ops.extend(self.variable_mgr.get_post_init_ops())

        if ((not self.single_session) and (not self.distributed_collective) and
                self.job_name and self.params.cross_replica_sync):
            # Ensure all workers execute variable_manager_init_ops before they start
            # executing the model.
            variable_manager_init_ops.append(
                self.add_sync_queues_and_barrier('init_ops_end_',
                    variable_manager_init_ops))

        local_var_init_op_group = tf.group(*variable_manager_init_ops,
            name='local_var_init_op_group')

        # with tf.name_scope('input_processing'):
        #     summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        #     print('since we care about summaries during preprocessing, we add {} summary ops to GraphInfo'.format(summaries))






        self.update_name_to_variables()

        return GraphInfo(
            input_producer_op=input_producer_op,
            enqueue_ops=enqueue_ops,
            fetches=fetches,
            execution_barrier=execution_barrier,
            global_step=global_step,
            local_var_init_op_group=local_var_init_op_group,
            mvav_op=variables_averages_op)


