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

import argparse
from collections import namedtuple
import math
import multiprocessing
import os
import re
import threading
import time

from absl import flags as absl_flags
import numpy as np

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cnn_util
import constants
import datasets
import flags
import variable_mgr
import variable_mgr_util
from cnn_util import log_fn
from models import model_config
from platforms import util as platforms_util
from google.protobuf import text_format
from tensorflow.compiler import xla
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline
from tensorflow.contrib.data.python.ops import prefetching_ops
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_util_impl
from tensorflow.python.framework import importer
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import nest


_DEFAULT_NUM_BATCHES = 100

# GraphInfo encapsulates the tensors/ops that we care about after building a
# graph. We use them to benchmark the graph.
GraphInfo = namedtuple(  # pylint: disable=invalid-name
    'GraphInfo',
    [
        # Ops that produce the input batches (before preprocessing).
        'input_producer_op',
        # Ops that adds the preprocessed images to the staging areas
        'enqueue_ops',
        # Fetches of sess.run()
        'fetches',
        # Op that performs synchronization in distributed mode
        'execution_barrier',
        # The global step variable
        'global_step',
        # Group of ops that perform per-device initialization work
        'local_var_init_op_group',

        'mvav_op'
    ])

# InputProcessingInfo contains various sources of inputs which will be later fed
# into the model. If synthetic data is used, all four fields are None.
InputProcessingInfo = namedtuple(
    'InputProcessingInfo',
    [
        # The first two fields are non-None iff datasets prefetching is not
        # used.

        # Ops that produce the input batches.
        'input_producer_op',
        # A list of StagingArea for each device.
        'input_producer_stages',

        # Input produced using FunctionBufferingResource. Non-None iff datasets
        # prefetching is used and --use_multi_device_iterator=False
        'function_buffering_resources',

        # Input produced using multi device iterator. Non-None iff datasets
        # prefetching is used and --use_multi_device_iterator=True
        'multi_device_iterator_input'
    ])

# TODO(reedwm): add upper_bound and lower_bound to appropriate integer and
# float flags, and change certain string flags to enum flags.

flags.DEFINE_string('model', 'trivial',
    'Name of the model to run, the list of supported models '
    'are defined in models/model.py')
# The code will first check if it's running under benchmarking mode
# or evaluation mode, depending on 'eval':
# Under the evaluation mode, this script will read a saved model,
#   and compute the accuracy of the model against a validation dataset.
#   Additional ops for accuracy and top_k predictors are only used under
#   this mode.
# Under the benchmarking mode, user can specify whether nor not to use
#   the forward-only option, which will only compute the loss function.
#   forward-only cannot be enabled with eval at the same time.
flags.DEFINE_boolean('eval', False, 'whether use eval or benchmarking')
flags.DEFINE_integer('eval_interval_secs', 0,
    'How often to run eval on saved checkpoints. Usually the '
    'same as save_model_secs from the corresponding training '
    'run. Pass 0 to eval only once.')
flags.DEFINE_boolean('forward_only', False,
    'whether use forward-only or training for benchmarking')
flags.DEFINE_boolean('freeze_when_forward_only', False,
    'whether to freeze the graph when in forward-only mode.')
flags.DEFINE_boolean('print_training_accuracy', False,
    'whether to calculate and print training accuracy during '
    'training')
flags.DEFINE_integer('batch_size', 0, 'batch size per compute device')
flags.DEFINE_integer('batch_group_size', 1,
    'number of groups of batches processed in the image '
    'producer.')
flags.DEFINE_integer('num_batches', None, 'number of batches to run, excluding '
                                          'warmup. Defaults to %d' % _DEFAULT_NUM_BATCHES)
flags.DEFINE_float('num_epochs', None,
    'number of epochs to run, excluding warmup. '
    'This and --num_batches cannot both be specified.')
flags.DEFINE_integer('num_warmup_batches', None,
    'number of batches to run before timing')
flags.DEFINE_integer('autotune_threshold', None,
    'The autotune threshold for the models')
flags.DEFINE_integer('num_gpus', 1, 'the number of GPUs to run on')
flags.DEFINE_string('gpu_indices', '', 'indices of worker GPUs in ring order')
flags.DEFINE_integer('display_every', 10,
    'Number of local steps after which progress is printed '
    'out')
flags.DEFINE_string('data_dir', None,
    'Path to dataset in TFRecord format (aka Example '
    'protobufs). If not specified, synthetic data will be '
    'used.')
flags.DEFINE_string('data_name', None,
    'Name of dataset: imagenet or cifar10. If not specified, '
    'it is automatically guessed based on data_dir.')
flags.DEFINE_string('resize_method', 'bilinear',
    'Method for resizing input images: crop, nearest, '
    'bilinear, bicubic, area, or round_robin. The `crop` mode '
    'requires source images to be at least as large as the '
    'network input size. The `round_robin` mode applies '
    'different resize methods based on position in a batch in '
    'a round-robin fashion. Other modes support any sizes and '
    'apply random bbox distortions before resizing (even with '
    'distortions=False).')
flags.DEFINE_boolean('distortions', True,
    'Enable/disable distortions during image preprocessing. '
    'These include bbox and color distortions.')
flags.DEFINE_boolean('use_datasets', True,
    'Enable use of datasets for input pipeline')
flags.DEFINE_string('input_preprocessor', 'default',
    'Name of input preprocessor. The list of supported input '
    'preprocessors are defined in preprocessing.py.')
flags.DEFINE_string('gpu_thread_mode', 'gpu_private',
    'Methods to assign GPU host work to threads. '
    'global: all GPUs and CPUs share the same global threads; '
    'gpu_private: a private threadpool for each GPU; '
    'gpu_shared: all GPUs share the same threadpool.')
flags.DEFINE_integer('per_gpu_thread_count', 0,
    'The number of threads to use for GPU. Only valid when '
    'gpu_thread_mode is not global.')
flags.DEFINE_boolean('hierarchical_copy', False,
    'Use hierarchical copies. Currently only optimized for '
    'use on a DGX-1 with 8 GPUs and may perform poorly on '
    'other hardware. Requires --num_gpus > 1, and only '
    'recommended when --num_gpus=8')
# TODO(hinsu): Support auto-detection of the network topology while still
# retaining the ability to specify a particular topology for debugging.
flags.DEFINE_enum(
    'network_topology', constants.NetworkTopology.DGX1,
    (constants.NetworkTopology.DGX1, constants.NetworkTopology.GCP_V100),
    'Network topology specifies the topology used to connect multiple devices. '
    'Network topology is used to decide the hierarchy to use for the '
    'hierarchical_copy.')
flags.DEFINE_integer('gradient_repacking', 0, 'Use gradient repacking. It'
                                              'currently only works with replicated mode. At the end of'
                                              'of each step, it repacks the gradients for more efficient'
                                              'cross-device transportation. A non-zero value specifies'
                                              'the number of split packs that will be formed.',
    lower_bound=0)
flags.DEFINE_boolean('compact_gradient_transfer', True, 'Compact gradient'
                                                        'as much as possible for cross-device transfer and '
                                                        'aggregation.')
flags.DEFINE_enum('variable_consistency', 'strong', ('strong', 'relaxed'),
    'The data consistency for trainable variables. With strong '
    'consistency, the variable always have the updates from '
    'previous step. With relaxed consistency, all the updates '
    'will eventually show up in the variables. Likely one step '
    'behind.')
flags.DEFINE_boolean('datasets_repeat_cached_sample', False,
    'Enable use of a special datasets pipeline that reads a '
    'single TFRecord into memory and repeats it infinitely '
    'many times. The purpose of this flag is to make it '
    'possible to write regression tests that are not '
    'bottlenecked by CNS throughput. '
    'Use datasets_use_caching to cache input data.')
flags.DEFINE_enum('local_parameter_device', 'gpu', ('cpu', 'gpu', 'CPU', 'GPU'),
    'Device to use as parameter server: cpu or gpu. For '
    'distributed training, it can affect where caching of '
    'variables happens.')
flags.DEFINE_enum('device', 'gpu', ('cpu', 'gpu', 'CPU', 'GPU'),
    'Device to use for computation: cpu or gpu')
flags.DEFINE_enum('data_format', 'NCHW', ('NHWC', 'NCHW'),
    'Data layout to use: NHWC (TF native) or NCHW (cuDNN '
    'native, requires GPU).')
flags.DEFINE_integer('num_intra_threads', None,
    'Number of threads to use for intra-op parallelism. If '
    'set to 0, the system will pick an appropriate number.')
flags.DEFINE_integer('num_inter_threads', 0,
    'Number of threads to use for inter-op parallelism. If '
    'set to 0, the system will pick an appropriate number.')
flags.DEFINE_string('trace_file', '',
    'Enable TensorFlow tracing and write trace to this file.')
flags.DEFINE_boolean('use_chrome_trace_format', True,
    'If True, the trace_file, if specified, will be in a '
    'Chrome trace format. If False, then it will be a '
    'StepStats raw proto.')
_NUM_STEPS_TO_PROFILE = 10
_NUM_OPS_TO_PRINT = 20
flags.DEFINE_string('tfprof_file', None,
    'If specified, write a tfprof ProfileProto to this file. '
    'The performance and other aspects of the model can then '
    'be analyzed with tfprof. See '
    'https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/command_line.md '  # pylint: disable=line-too-long
    'for more info on how to do this. The first %d steps '
    'are profiled. Additionally, the top %d most time '
    'consuming ops will be printed.\n'
    'Note: profiling with tfprof is very slow, but most of the '
    'overhead is spent between steps. So, profiling results '
    'are more accurate than the slowdown would suggest.' %
    (_NUM_STEPS_TO_PROFILE, _NUM_OPS_TO_PRINT))
flags.DEFINE_string('graph_file', None,
    'Write the model\'s graph definition to this file. '
    'Defaults to binary format unless filename ends in "txt".')
flags.DEFINE_string('partitioned_graph_file_prefix', None,
    'If specified, after the graph has been partitioned and '
    'optimized, write out each partitioned graph to a file '
    'with the given prefix.')
flags.DEFINE_enum('optimizer', 'sgd', ('momentum', 'sgd', 'rmsprop', 'adam'),
    'Optimizer to use')
flags.DEFINE_float('init_learning_rate', None,
    'Initial learning rate for training.')
flags.DEFINE_string('piecewise_learning_rate_schedule', None,
    'Specifies a piecewise learning rate schedule based on the '
    'number of epochs. This is the form LR0;E1;LR1;...;En;LRn, '
    'where each LRi is a learning rate and each Ei is an epoch '
    'indexed from 0. The learning rate is LRi if the '
    'E(i-1) <= current_epoch < Ei. For example, if this '
    'paramater is 0.3;10;0.2;25;0.1, the learning rate is 0.3 '
    'for the first 10 epochs, then is 0.2 for the next 15 '
    'epochs, then is 0.1 until training ends.')
flags.DEFINE_float('num_epochs_per_decay', 0,
    'Steps after which learning rate decays. If 0, the learning '
    'rate does not decay.')
flags.DEFINE_float('learning_rate_decay_factor', 0,
    'Learning rate decay factor. Decay by this factor every '
    '`num_epochs_per_decay` epochs. If 0, learning rate does '
    'not decay.')
flags.DEFINE_float('num_learning_rate_warmup_epochs', 0,
    'Slowly increase to the initial learning rate in the first '
    'num_learning_rate_warmup_epochs linearly.')
flags.DEFINE_float('minimum_learning_rate', 0,
    'The minimum learning rate. The learning rate will '
    'never decay past this value. Requires `learning_rate`, '
    '`num_epochs_per_decay` and `learning_rate_decay_factor` to '
    'be set.')
flags.DEFINE_float('momentum', 0.9, 'Momentum for training.')
flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum in RMSProp.')
flags.DEFINE_float('rmsprop_epsilon', 1.0, 'Epsilon term for RMSProp.')
flags.DEFINE_float('adam_beta1', 0.9, 'Beta2 term for the Adam optimizer')
flags.DEFINE_float('adam_beta2', 0.999, 'Beta2 term for the Adam optimizer')
flags.DEFINE_float('adam_epsilon', 1e-8, 'Epsilon term for the Adam optimizer')
flags.DEFINE_float('gradient_clip', None,
    'Gradient clipping magnitude. Disabled by default.')
flags.DEFINE_float('weight_decay', 0.00004,
    'Weight decay factor for training.')
flags.DEFINE_float('gpu_memory_frac_for_testing', 0,
    'If non-zero, the fraction of GPU memory that will be used. '
    'Useful for testing the benchmark script, as this allows '
    'distributed mode to be run on a single machine. For '
    'example, if there are two tasks, each can be allocated '
    '~40 percent of the memory on a single machine',
    lower_bound=0., upper_bound=1.)
flags.DEFINE_boolean('use_unified_memory', False,
    'If True, allocate unified memory enabling larger models '
    'to fit in available device RAM.')
flags.DEFINE_boolean('use_tf_layers', True,
    'If True, use tf.layers for neural network layers. This '
    'should not affect performance or accuracy in any way.')
flags.DEFINE_integer('tf_random_seed', 1234,
    'The TensorFlow random seed. Useful for debugging NaNs, '
    'as this can be set to various values to see if the NaNs '
    'depend on the seed.')
flags.DEFINE_string('debugger', None,
    'If set, use the TensorFlow debugger. If set to "cli", use '
    'the local CLI debugger. Otherwise, this must be in the '
    'form hostname:port (e.g., localhost:7007) in which case '
    'the experimental TensorBoard debugger will be used')
flags.DEFINE_boolean('use_python32_barrier', False,
    'When on, use threading.Barrier at Python 3.2.')

flags.DEFINE_boolean('datasets_use_prefetch', True,
    'Enable use of prefetched datasets for input pipeline. '
    'This option is meaningless if use_datasets=False.')
flags.DEFINE_integer('datasets_prefetch_buffer_size', 1,
    'Prefetching op buffer size per compute device.')
flags.DEFINE_integer('datasets_num_private_threads', None,
    'Number of threads for a private threadpool created for '
    'all datasets computation. By default, we pick an '
    'appropriate number. If set to 0, we use the default '
    'tf-Compute threads for dataset operations.')
flags.DEFINE_boolean('datasets_use_caching', False,
    'Cache the compressed input data in memory. This improves '
    'the data input performance, at the cost of additional '
    'memory.')
flags.DEFINE_integer('datasets_parallel_interleave_cycle_length', None,
    'Number of parallel file readers interleaving input data.')
flags.DEFINE_boolean('datasets_sloppy_parallel_interleave', False,
    'Allow parallel interleave to depart from deterministic '
    'ordering, by temporarily skipping over files whose '
    'elements are not readily available. This can increase '
    'througput in particular in the presence of stragglers.')

flags.DEFINE_boolean(
    'use_multi_device_iterator', True,
    'If true, we use the MultiDeviceIterator for prefetching, '
    'which deterministically prefetches the data onto the '
    'various GPUs')
flags.DEFINE_integer(
    'multi_device_iterator_max_buffer_size', 1,
    'Configuration parameter for the MultiDeviceIterator that '
    ' specifies the host side buffer size for each device.')

# Performance tuning parameters.
flags.DEFINE_boolean('winograd_nonfused', True,
    'Enable/disable using the Winograd non-fused algorithms.')
flags.DEFINE_boolean(
    'batchnorm_persistent', True,
    'Enable/disable using the CUDNN_BATCHNORM_SPATIAL_PERSISTENT '
    'mode for batchnorm.')
flags.DEFINE_boolean('sync_on_finish', False,
    'Enable/disable whether the devices are synced after each '
    'step.')
flags.DEFINE_boolean('staged_vars', False,
    'whether the variables are staged from the main '
    'computation')
flags.DEFINE_boolean('force_gpu_compatible', False,
    'whether to enable force_gpu_compatible in GPU_Options')
flags.DEFINE_boolean('allow_growth', None,
    'whether to enable allow_growth in GPU_Options')
flags.DEFINE_boolean('xla', False, 'whether to enable XLA auto-jit compilation')
flags.DEFINE_boolean('xla_compile', False,
    'Enable xla to compile the graph. Uncompilable ops will '
    'result in fatal errors.')
flags.DEFINE_boolean('fuse_decode_and_crop', True,
    'Fuse decode_and_crop for image preprocessing.')
flags.DEFINE_boolean('distort_color_in_yiq', True,
    'Distort color of input images in YIQ space.')
flags.DEFINE_boolean('enable_optimizations', True,
    'Whether to enable grappler and other optimizations.')
flags.DEFINE_string('rewriter_config', None,
    'Config for graph optimizers, described as a '
    'RewriterConfig proto buffer.')
flags.DEFINE_enum('loss_type_to_report', 'total_loss',
    ('base_loss', 'total_loss'),
    'Which type of loss to output and to write summaries for. '
    'The total loss includes L2 loss while the base loss does '
    'not. Note that the total loss is always used while '
    'computing gradients during training if weight_decay > 0, '
    'but explicitly computing the total loss, instead of just '
    'computing its gradients, can have a performance impact.')
flags.DEFINE_boolean('single_l2_loss_op', False,
    'If True, instead of using an L2 loss op per variable, '
    'concatenate the variables into a single tensor and do a '
    'single L2 loss on the concatenated tensor.')
flags.DEFINE_boolean('use_resource_vars', False,
    'Use resource variables instead of normal variables. '
    'Resource variables are slower, but this option is useful '
    'for debugging their performance.')
# Performance tuning specific to MKL.
flags.DEFINE_boolean('mkl', False, 'If true, set MKL environment variables.')
flags.DEFINE_integer('kmp_blocktime', 0,
    'The time, in milliseconds, that a thread should wait, '
    'after completing the execution of a parallel region, '
    'before sleeping')
flags.DEFINE_string('kmp_affinity', 'granularity=fine,verbose,compact,1,0',
    'Restricts execution of certain threads (virtual execution '
    'units) to a subset of the physical processing units in a '
    'multiprocessor computer.')
flags.DEFINE_integer('kmp_settings', 1,
    'If set to 1, MKL settings will be printed.')

# fp16 parameters. If use_fp16=False, no other fp16 parameters apply.
flags.DEFINE_boolean('use_fp16', False,
    'Use 16-bit floats for certain tensors instead of 32-bit '
    'floats. This is currently experimental.')
# TODO(reedwm): The default loss scale of 128 causes most models to diverge
# on the second step with synthetic data. Changing the tf.set_random_seed
# call to tf.set_random_seed(1235) or most other seed values causes the
# issue not to occur.
flags.DEFINE_float('fp16_loss_scale', None,
    'If fp16 is enabled, the loss is multiplied by this amount '
    'right before gradients are computed, then each gradient '
    'is divided by this amount. Mathematically, this has no '
    'effect, but it helps avoid fp16 underflow. Set to 1 to '
    'effectively disable.')
flags.DEFINE_boolean('fp16_vars', False,
    'If fp16 is enabled, also use fp16 for variables. If '
    'False, the variables are stored in fp32 and casted to '
    'fp16 when retrieved.  Recommended to leave as False.')
flags.DEFINE_boolean('fp16_enable_auto_loss_scale', False,
    'If True and use_fp16 is True, automatically adjust the '
    'loss scale during training.')
flags.DEFINE_integer('fp16_inc_loss_scale_every_n', 1000,
    'If fp16 is enabled and fp16_enable_auto_loss_scale is '
    'True, increase the loss scale every n steps.')

# The method for managing variables:
#   parameter_server: variables are stored on a parameter server that holds
#       the master copy of the variable. In local execution, a local device
#       acts as the parameter server for each variable; in distributed
#       execution, the parameter servers are separate processes in the
#       cluster.
#       For each step, each tower gets a copy of the variables from the
#       parameter server, and sends its gradients to the param server.
#   replicated: each GPU has its own copy of the variables. To apply
#       gradients, an all_reduce algorithm or or regular cross-device
#       aggregation is used to replicate the combined gradients to all
#       towers (depending on all_reduce_spec parameter setting).
#   independent: each GPU has its own copy of the variables, and gradients
#       are not shared between towers. This can be used to check performance
#       when no data is moved between GPUs.
#   distributed_replicated: Distributed training only. Each GPU has a copy
#       of the variables, and updates its copy after the parameter servers
#       are all updated with the gradients from all servers. Only works with
#       cross_replica_sync=true. Unlike 'replicated', currently never uses
#       nccl all-reduce for replicating within a server.
#   distributed_all_reduce: Distributed training where all replicas run
#       in a single session, using all-reduce to mutally reduce the
#       gradients.  Uses no parameter servers.  When there is only one
#       worker, this is the same as replicated.
#   collective_all_reduce: Distributed training where all replicas run
#       independepently except for variable initialization and for
#       gradient reduction which is done via collective all-reduce.
#       NOTE: collective_all_reduce in conjunction with use_fp16 can
#       lead to NaNs in some models (resnet50).  TODO(tucker): fix it.
#   horovod: Distributed training using Horovod library. Runs workers using
#       an MPI framework (e.g. Open MPI). Each worker runs training on
#       single GPU, and averages gradients using NCCL or MPI all-reduce.
#       See https://github.com/uber/horovod for more details.
flags.DEFINE_enum('variable_update', 'parameter_server',
    ('parameter_server', 'replicated', 'distributed_replicated',
     'independent', 'distributed_all_reduce',
     'collective_all_reduce', 'horovod'),
    'The method for managing variables: parameter_server, '
    'replicated, distributed_replicated, independent, '
    'distributed_all_reduce, collective_all_reduce, horovod')
flags.DEFINE_string('all_reduce_spec', None,
    'A specification of the all_reduce algorithm to be used '
    'for reducing gradients.  For more details, see '
    'parse_all_reduce_spec in variable_mgr.py.  An '
    'all_reduce_spec has BNF form:\n'
    'int ::= positive whole number\n'
    'g_int ::= int[KkMGT]?\n'
    'alg_spec ::= alg | alg#int\n'
    'range_spec ::= alg_spec | alg_spec/alg_spec\n'
    'spec ::= range_spec | range_spec:g_int:range_spec\n'
    'NOTE: not all syntactically correct constructs are '
    'supported.\n\n'
    'Examples:\n '
    '"xring" == use one global ring reduction for all '
    'tensors\n'
    '"pscpu" == use CPU at worker 0 to reduce all tensors\n'
    '"nccl" == use NCCL to locally reduce all tensors.  '
    'Limited to 1 worker.\n'
    '"nccl/xring" == locally (to one worker) reduce values '
    'using NCCL then ring reduce across workers.\n'
    '"pscpu:32k:xring" == use pscpu algorithm for tensors of '
    'size up to 32kB, then xring for larger tensors.')

# If variable_update==distributed_all_reduce then it may be advantageous
# to aggregate small tensors into one prior to reduction.  These parameters
# control that aggregation.
flags.DEFINE_integer('agg_small_grads_max_bytes', 0,
    'If > 0, try to aggregate tensors of less than this '
    'number of bytes prior to all-reduce.')
flags.DEFINE_integer('agg_small_grads_max_group', 10,
    'When aggregating small tensors for all-reduce do not '
    'aggregate more than this many into one new tensor.')
flags.DEFINE_integer('allreduce_merge_scope', 1,
    'Establish a name scope around this many '
    'gradients prior to creating the all-reduce operations. '
    'It may affect the ability of the backend to merge '
    'parallel ops.')

# Distributed training parameters.
flags.DEFINE_enum('job_name', '', ('ps', 'worker', 'controller', ''),
    'One of "ps", "worker", "controller", "".  Empty for local '
    'training')
flags.DEFINE_string('ps_hosts', '', 'Comma-separated list of target hosts')
flags.DEFINE_string('worker_hosts', '', 'Comma-separated list of target hosts')
flags.DEFINE_string('controller_host', None, 'optional controller host')
flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
flags.DEFINE_string('server_protocol', 'grpc', 'protocol for servers')
flags.DEFINE_boolean('cross_replica_sync', True, '')
flags.DEFINE_string('horovod_device', '', 'Device to do Horovod all-reduce on: '
                                          'empty (default), cpu or gpu. Default with utilize GPU if '
                                          'Horovod was compiled with the HOROVOD_GPU_ALLREDUCE '
                                          'option, and CPU otherwise.')

# Summary and Save & load checkpoints.
flags.DEFINE_integer('summary_verbosity', 0, 'Verbosity level for summary ops. '
                                             'level 0: disable any summary.\n'
                                             'level 1: small and fast ops, e.g.: learning_rate, '
                                             'total_loss.\n'
                                             'level 2: medium-cost ops, e.g. histogram of all '
                                             'gradients.\n'
                                             'level 3: expensive ops: images and histogram of each '
                                             'gradient.\n')
flags.DEFINE_integer('save_summaries_steps', 0,
    'How often to save summaries for trained models. Pass 0 '
    'to disable summaries.')
flags.DEFINE_integer('save_model_secs', 0,
    'How often to save trained models. Pass 0 to disable '
    'saving checkpoints every N seconds. A checkpoint is '
    'saved after training completes regardless of this '
    'option.')
flags.DEFINE_integer('save_model_steps', None,
    'How often to save trained models. If specified, '
    'save_model_secs must not be specified.')
flags.DEFINE_integer('max_ckpts_to_keep', 5,
    'Max number of checkpoints to keep.')
flags.DEFINE_string('train_dir', None,
    'Path to session checkpoints. Pass None to disable saving '
    'checkpoint at the end.')
flags.DEFINE_string('eval_dir', '/tmp/tf_cnn_benchmarks/eval',
    'Directory where to write eval event logs.')
flags.DEFINE_string('backbone_model_path', None,
    'Path to pretrained backbone model checkpoint. Pass None '
    'if not using a backbone model.')
flags.DEFINE_enum('trt_mode', '', ['', 'FP32', 'FP16', 'INT8'],
    'If this is specified in forward_only mode and '
    'freeze_when_forward_only is set to True, use TensorRT to '
    'optimize the graph before execution.')
flags.DEFINE_integer('trt_max_workspace_size_bytes', 4 << 30,
    'Max workspace size bytes used by the TensorRT optimizer.')

# Benchmark logging for model garden metric
flags.DEFINE_string('benchmark_log_dir', None,
    'The directory to place the log files containing the '
    'results of benchmark. The logs are created by '
    'BenchmarkFileLogger. Requires the root of the Tensorflow '
    'models repository to be in $PYTHTONPATH.')
flags.DEFINE_string('benchmark_test_id', None,
    'The unique test ID of the benchmark run. It could be the '
    'combination of key parameters. It is hardware independent '
    'and could be used compare the performance between '
    'different test runs. This flag is designed for human '
    'consumption, and does not have any impact within the '
    'system.')

platforms_util.define_platform_params()


class GlobalStepWatcher(threading.Thread):
    """A helper class for global_step.

    Polls for changes in the global_step of the model, and finishes when the
    number of steps for the global run are done.
    """

    def __init__(self, sess, global_step_op, start_at_global_step,
                 end_at_global_step):
        threading.Thread.__init__(self)
        self.sess = sess
        self.global_step_op = global_step_op
        self.start_at_global_step = start_at_global_step
        self.end_at_global_step = end_at_global_step

        self.start_time = 0
        self.start_step = 0
        self.finish_time = 0
        self.finish_step = 0

    def run(self):
        while self.finish_time == 0:
            time.sleep(.25)
            global_step_val, = self.sess.run([self.global_step_op])
            if self.start_time == 0 and global_step_val >= self.start_at_global_step:
                # Use tf.logging.info instead of log_fn, since print (which is log_fn)
                # is not thread safe and may interleave the outputs from two parallel
                # calls to print, which can break tests.
                tf.logging.info('Starting real work at step %s at time %s' %
                                (global_step_val, time.ctime()))
                self.start_time = time.time()
                self.start_step = global_step_val
            if self.finish_time == 0 and global_step_val >= self.end_at_global_step:
                tf.logging.info('Finishing real work at step %s at time %s' %
                                (global_step_val, time.ctime()))
                self.finish_time = time.time()
                self.finish_step = global_step_val

    def done(self):
        return self.finish_time > 0

    def num_steps(self):
        return self.finish_step - self.start_step

    def elapsed_time(self):
        return self.finish_time - self.start_time


class CheckpointNotFoundException(Exception):
    pass


def create_config_proto(params):
    """Returns session config proto.

    Args:
      params: Params tuple, typically created by make_params or
              make_params_from_flags.
    """
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    if params.num_intra_threads is None:
        if params.device == 'gpu':
            config.intra_op_parallelism_threads = 1
    else:
        config.intra_op_parallelism_threads = params.num_intra_threads
    config.inter_op_parallelism_threads = params.num_inter_threads
    config.experimental.collective_group_leader = '/job:worker/replica:0/task:0'
    config.gpu_options.force_gpu_compatible = params.force_gpu_compatible
    if params.allow_growth is not None:
        config.gpu_options.allow_growth = params.allow_growth
    if params.gpu_memory_frac_for_testing > 0:
        config.gpu_options.per_process_gpu_memory_fraction = (
            params.gpu_memory_frac_for_testing)
    if params.use_unified_memory:
        config.gpu_options.experimental.use_unified_memory = True
    if params.xla:
        config.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1)
        # TODO(b/117324590): Re-enable PinToHostOptimizer when b/117324590 is fixed.
        # Currently we have to disable PinToHostOptimizer w/ XLA since it causes
        # OOM/perf cliffs.
        config.graph_options.rewrite_options.pin_to_host_optimization = (
            rewriter_config_pb2.RewriterConfig.OFF)
    if params.rewriter_config:
        rewriter_config = rewriter_config_pb2.RewriterConfig()
        text_format.Merge(params.rewriter_config, rewriter_config)
        config.graph_options.rewrite_options.CopyFrom(rewriter_config)
    elif not params.enable_optimizations:
        off = rewriter_config_pb2.RewriterConfig.OFF
        config.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0
        rewrite_options = config.graph_options.rewrite_options
        rewrite_options.layout_optimizer = off
        rewrite_options.constant_folding = off
        rewrite_options.shape_optimization = off
        rewrite_options.remapping = off
        rewrite_options.arithmetic_optimization = off
        rewrite_options.dependency_optimization = off
        rewrite_options.loop_optimization = off
        rewrite_options.function_optimization = off
        rewrite_options.debug_stripper = off
        rewrite_options.disable_model_pruning = True
        rewrite_options.scoped_allocator_optimization = off
        rewrite_options.memory_optimization = (
            rewriter_config_pb2.RewriterConfig.NO_MEM_OPT)
        rewrite_options.pin_to_host_optimization = off
    elif params.variable_update == 'collective_all_reduce':
        rewrite_options = config.graph_options.rewrite_options
        rewrite_options.scoped_allocator_optimization = (
            rewriter_config_pb2.RewriterConfig.ON)
        rewrite_options.scoped_allocator_opts.enable_op.append('CollectiveReduce')
    if params.variable_update == 'horovod':
        import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
        config.gpu_options.visible_device_list = str(hvd.local_rank())

    return config


def get_mode_from_params(params):
    """Returns the mode in which this script is running.

    Args:
      params: Params tuple, typically created by make_params or
              make_params_from_flags.
    Raises:
      ValueError: Unsupported params settings.
    """
    if params.forward_only and params.eval:
        raise ValueError('Only one of forward_only and eval parameters is true')

    if params.eval:
        return 'evaluation'
    if params.forward_only:
        return 'forward-only'
    return 'training'


# How many digits to show for the loss and accuracies during training.
LOSS_AND_ACCURACY_DIGITS_TO_SHOW = 6


def benchmark_one_step(sess,
                       fetches,
                       step,
                       batch_size,
                       step_train_times,
                       trace_filename,
                       partitioned_graph_file_prefix,
                       profiler,
                       image_producer,
                       params,
                       summary_op=None,
                       show_images_per_sec=True,
                       benchmark_logger=None,
                       collective_graph_key=0,
                            track_mvav_op=None,
                       extra_feed_dict=None,
                       extra_fetch_dict=None):
    """Advance one step of benchmarking."""
    should_profile = profiler and 0 <= step < _NUM_STEPS_TO_PROFILE
    need_options_and_metadata = (
        should_profile or collective_graph_key > 0 or
        ((trace_filename or partitioned_graph_file_prefix) and step == -2)
    )
    if need_options_and_metadata:
        run_options = tf.RunOptions()
        if (trace_filename and step == -2) or should_profile:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        if partitioned_graph_file_prefix and step == -2:
            run_options.output_partition_graphs = True
        if collective_graph_key > 0:
            run_options.experimental.collective_graph_key = collective_graph_key
        run_metadata = tf.RunMetadata()
    else:
        run_options = None
        run_metadata = None

    to_fetch = {'origin': fetches}
    if summary_op is not None:
        to_fetch['summary'] = summary_op
    if track_mvav_op is not None:
        to_fetch['mvav'] = track_mvav_op
    if extra_fetch_dict is not None:
        to_fetch['extra'] = extra_fetch_dict

    start_time = time.time()


    all_fetched_results = sess.run(to_fetch, options=run_options, run_metadata=run_metadata, feed_dict=extra_feed_dict)
    results = all_fetched_results['origin']

    summary_str = all_fetched_results.get('summary', None)


    if not params.forward_only:
        lossval = results['average_loss']
    else:
        lossval = 0.
    if image_producer is not None:
        image_producer.notify_image_consumption()
    train_time = time.time() - start_time
    step_train_times.append(train_time)
    if (show_images_per_sec and step >= 0 and
            (step == 0 or (step + 1) % params.display_every == 0)):
        speed_mean, speed_uncertainty, speed_jitter = get_perf_timing(
            batch_size, step_train_times)
        log_str = '%i\t%s\t%.*f' % (
            step + 1,
            get_perf_timing_str(speed_mean, speed_uncertainty, speed_jitter),
            LOSS_AND_ACCURACY_DIGITS_TO_SHOW, lossval)
        if 'top_1_accuracy' in results:
            log_str += '\t%.*f\t%.*f' % (
                LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_1_accuracy'],
                LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_5_accuracy'])
        log_fn(log_str)
        if benchmark_logger:
            benchmark_logger.log_metric(
                'current_examples_per_sec', speed_mean, global_step=step + 1)
            if 'top_1_accuracy' in results:
                benchmark_logger.log_metric(
                    'top_1_accuracy', results['top_1_accuracy'], global_step=step + 1)
                benchmark_logger.log_metric(
                    'top_5_accuracy', results['top_5_accuracy'], global_step=step + 1)
    if need_options_and_metadata:
        if should_profile:
            profiler.add_step(step, run_metadata)
        if trace_filename and step == -2:
            log_fn('Dumping trace to %s' % trace_filename)
            trace_dir = os.path.dirname(trace_filename)
            if not gfile.Exists(trace_dir):
                gfile.MakeDirs(trace_dir)
            with gfile.Open(trace_filename, 'w') as trace_file:
                if params.use_chrome_trace_format:
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
                else:
                    trace_file.write(str(run_metadata.step_stats))
        if partitioned_graph_file_prefix and step == -2:
            path, filename = os.path.split(partitioned_graph_file_prefix)
            if '.' in filename:
                base_filename, ext = filename.rsplit('.', 1)
                ext = '.' + ext
            else:
                base_filename, ext = filename, ''
            as_text = filename.endswith('txt')
            for graph_def in run_metadata.partition_graphs:
                device = graph_def.node[0].device.replace('/', '_').replace(':', '_')
                graph_filename = '%s%s%s' % (base_filename, device, ext)
                log_fn('Writing partitioned GraphDef as %s to %s' % (
                    'text' if as_text else 'binary',
                    os.path.join(path, graph_filename)))
                tf.train.write_graph(graph_def, path, graph_filename, as_text)
    return summary_str, lossval, all_fetched_results.get('extra', None)


def get_perf_timing_str(speed_mean, speed_uncertainty, speed_jitter, scale=1):
    if scale == 1:
        # TODO(laigd): rename 'images' to maybe 'inputs', same below.
        return ('images/sec: %.1f +/- %.1f (jitter = %.1f)' %
                (speed_mean, speed_uncertainty, speed_jitter))
    else:
        return 'images/sec: %.1f' % speed_mean


def get_perf_timing(batch_size, step_train_times, scale=1):
    times = np.array(step_train_times)
    speeds = batch_size / times
    speed_mean = scale * batch_size / np.mean(times)
    speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
    speed_jitter = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
    return speed_mean, speed_uncertainty, speed_jitter


def load_checkpoint(saver, sess, ckpt_dir):
    """Loads checkpoint from provided directory or full path.

    Args:
      saver: Saver used to restore the checkpoint.
      sess: TensorFlow session.
      ckpt_dir: Path to a folder of checkpoints or full path to a checkpoint.

    Returns:
      Global step.
    """
    model_checkpoint_path = _get_checkpoint_to_load(ckpt_dir)
    global_step = model_checkpoint_path.split('/')[-1].split('-')[-1]
    if not global_step.isdigit():
        global_step = 0
    else:
        global_step = int(global_step)
    saver.restore(sess, model_checkpoint_path)
    log_fn('Successfully loaded model from %s.' % model_checkpoint_path)
    return global_step, model_checkpoint_path


def _get_checkpoint_to_load(ckpt_dir):
    """Returns which checkpoint to load.

    Args:
      ckpt_dir: Path to a folder of checkpoints or full path to a checkpoint.

    Returns:
      Full path to checkpoint to load.

    Raises:
      CheckpointNotFoundException: If checkpoint is not found.
    """
    p = re.compile(r'ckpt-\d+$')
    if p.search(ckpt_dir):
        model_checkpoint_path = ckpt_dir
    else:
        # Finds latest checkpoint in directory provided
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            model_checkpoint_path = ckpt.model_checkpoint_path
        else:
            raise CheckpointNotFoundException('No checkpoint file found in dir:{}'.
                format(ckpt_dir))
    return model_checkpoint_path


# Params are passed to BenchmarkCNN's constructor. Params is a map from name
# to value, with one field per key in flags.param_specs.
#
# Call make_params() or make_params_from_flags() below to construct a Params
# tuple with default values from flags.param_specs, rather than constructing
# Params directly.
Params = namedtuple('Params', flags.param_specs.keys())  # pylint: disable=invalid-name


def validate_params(params):
    """Validates that the Params tuple had valid values.

    When command-line flags are defined for each ParamSpec by calling
    flags.define_flags(), calling this function is unnecessary because absl
    already does flag validation. Otherwise, this function should be called.

    Args:
       params: A Params tuple.
    Raises:
      ValueError: An element of params had an invalid value.
    """
    for name, value in params._asdict().items():
        param_spec = flags.param_specs[name]
        if param_spec.flag_type in ('integer', 'float'):
            if (param_spec.kwargs['lower_bound'] is not None and
                        value < param_spec.kwargs['lower_bound']):
                raise ValueError('Param %s value of %s is lower than the lower bound '
                                 'of %s' %
                                 (name, value, param_spec.kwargs['lower_bound']))
            if (param_spec.kwargs['upper_bound'] is not None and
                        param_spec.kwargs['upper_bound'] < value):
                raise ValueError('Param %s value of %s is higher than the upper bound '
                                 'of %s' %
                                 (name, value, param_spec.kwargs['upper_bound']))
        elif (param_spec.flag_type == 'enum' and
                      value not in param_spec.kwargs['enum_values']):
            raise ValueError('Param %s of value %s is not in %s' %
                             (name, value, param_spec.kwargs['enum_values']))


def make_params(**kwargs):
    """Create a Params tuple for BenchmarkCNN from kwargs.

    Default values are filled in from flags.param_specs.

    Args:
      **kwargs: kwarg values will override the default values.
    Returns:
      Params namedtuple for constructing BenchmarkCNN.
    """
    # Create a (name: default_value) map from flags.param_specs.
    default_kwargs = {
        name: flags.param_specs[name].default_value
        for name in flags.param_specs
    }
    params = Params(**default_kwargs)._replace(**kwargs)
    validate_params(params)
    return params


def make_params_from_flags():
    """Create a Params tuple for BenchmarkCNN from absl_flags.FLAGS.

    Returns:
      Params namedtuple for constructing BenchmarkCNN.
    """
    # Collect (name: value) pairs for absl_flags.FLAGS with matching names in
    # flags.param_specs.
    flag_values = {name: getattr(absl_flags.FLAGS, name)
                   for name in flags.param_specs.keys()}
    return Params(**flag_values)


def get_num_batches_and_epochs(params, batch_size, num_examples_per_epoch):
    """Returns the number of batches and epochs to run for.

    Args:
      params: Params tuple, typically created by make_params or
        make_params_from_flags.
      batch_size: The number of images per step.
      num_examples_per_epoch: The number of images in a single epoch.

    Returns:
      num_batches: The number of batches to run for.
      num_epochs: The number of epochs to run for. This might be slightly
        smaller than params.num_epochs if specified, because the number of batches
        must be an integer.

    Raises:
      ValueError: Invalid or unsupported params.
    """
    if params.num_batches and params.num_epochs:
        raise ValueError('At most one of --num_batches and --num_epochs may be '
                         'specified.')
    if params.num_epochs:
        num_batches = int(float(params.num_epochs) * num_examples_per_epoch /
                          batch_size)
    else:
        num_batches = params.num_batches or _DEFAULT_NUM_BATCHES
    num_epochs = num_batches * batch_size / float(num_examples_per_epoch)
    return (num_batches, num_epochs)


def get_piecewise_learning_rate(piecewise_learning_rate_schedule,
                                global_step, num_batches_per_epoch):
    """Returns a piecewise learning rate tensor.

    Args:
      piecewise_learning_rate_schedule: The --piecewise_learning_rate_schedule
        parameter
      global_step: Scalar tensor representing the global step.
      num_batches_per_epoch: float indicating the number of batches per epoch.

    Returns:
      A scalar float tensor, representing the learning rate.

    Raises:
      ValueError: piecewise_learning_rate_schedule is not formatted correctly.
    """
    pieces = piecewise_learning_rate_schedule.split(';')
    if len(pieces) % 2 == 0:
        raise ValueError('--piecewise_learning_rate_schedule must have an odd '
                         'number of components')
    values = []
    boundaries = []
    for i, piece in enumerate(pieces):
        if i % 2 == 0:
            try:
                values.append(float(piece))
            except ValueError:
                raise ValueError('Invalid learning rate: ' + piece)
        else:
            try:
                boundaries.append(int(int(piece) * num_batches_per_epoch) - 1)
            except ValueError:
                raise ValueError('Invalid epoch: ' + piece)
    return tf.train.piecewise_constant(global_step, boundaries, values,
        name='piecewise_learning_rate'), boundaries


def get_learning_rate(params, global_step, num_examples_per_epoch, model,
                      batch_size):
    """Returns a learning rate tensor based on global_step.

    Args:
      params: Params tuple, typically created by make_params or
        make_params_from_flags.
      global_step: Scalar tensor representing the global step.
      num_examples_per_epoch: The number of examples per epoch.
      model: The model.Model object to obtain the default learning rate from if no
        learning rate is specified.
      batch_size: Number of examples per step

    Returns:
      A scalar float tensor, representing the learning rate. When evaluated, the
      learning rate depends on the current value of global_step.

    Raises:
      ValueError: Invalid or unsupported params.
    """
    with tf.name_scope('learning_rate'):
        num_batches_per_epoch = (float(num_examples_per_epoch) / batch_size)

        if params.piecewise_learning_rate_schedule:
            if (params.init_learning_rate or params.learning_rate_decay_factor or
                    params.minimum_learning_rate or params.num_epochs_per_decay):
                raise ValueError('No other learning rate-related flags can be '
                                 'specified if --piecewise_learning_rate_schedule is '
                                 'specified')
            learning_rate, boundaries = get_piecewise_learning_rate(
                params.piecewise_learning_rate_schedule,
                global_step, num_batches_per_epoch)


        elif params.init_learning_rate:
            learning_rate = params.init_learning_rate
            if (params.num_epochs_per_decay > 0 and
                        params.learning_rate_decay_factor > 0):
                decay_steps = int(num_batches_per_epoch * params.num_epochs_per_decay)

                # Decay the learning rate exponentially based on the number of steps.
                learning_rate = tf.train.exponential_decay(
                    params.init_learning_rate,
                    global_step,
                    decay_steps,
                    params.learning_rate_decay_factor,
                    staircase=True)

                if params.minimum_learning_rate != 0.:
                    learning_rate = tf.maximum(learning_rate,
                        params.minimum_learning_rate)

            boundaries = None
        else:
            learning_rate = model.get_learning_rate(global_step, batch_size)
            boundaries = None

        if params.num_learning_rate_warmup_epochs > 0 and (
                    params.init_learning_rate or params.piecewise_learning_rate_schedule):
            warmup_steps = int(num_batches_per_epoch *
                               params.num_learning_rate_warmup_epochs)
            init_lr = (params.init_learning_rate or
                       float(params.piecewise_learning_rate_schedule.split(';')[0]))
            warmup_lr = init_lr * tf.cast(global_step, tf.float32) / tf.cast(
                warmup_steps, tf.float32)
            learning_rate = tf.cond(global_step < warmup_steps,
                lambda: warmup_lr, lambda: learning_rate)

    return learning_rate, boundaries


def get_optimizer(params, learning_rate):
    """Returns the optimizer that should be used based on params."""
    if params.optimizer == 'momentum':
        opt = tf.train.MomentumOptimizer(
            learning_rate, params.momentum, use_nesterov=True)
    elif params.optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    elif params.optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(
            learning_rate,
            params.rmsprop_decay,
            momentum=params.rmsprop_momentum,
            epsilon=params.rmsprop_epsilon)
    elif params.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(learning_rate, params.adam_beta1,
            params.adam_beta2, params.adam_epsilon)
    else:
        raise ValueError('Optimizer "%s" was not recognized',
            params.optimizer)
    return opt


def generate_tfprof_profile(profiler, tfprof_file):
    """Generates a tfprof profile, writing it to a file and printing top ops.

    Args:
      profiler: A tf.profiler.Profiler. `profiler.add_step` must have already been
        called.
      tfprof_file: The filename to write the ProfileProto to.
    """
    profile_proto = profiler.serialize_to_string()
    log_fn('Dumping ProfileProto to %s' % tfprof_file)
    with gfile.Open(tfprof_file, 'wb') as f:
        f.write(profile_proto)

    # Print out the execution times of the top operations. Note this
    # information can also be obtained with the dumped ProfileProto, but
    # printing it means tfprof doesn't have to be used if all the user wants
    # is the top ops.
    options = tf.profiler.ProfileOptionBuilder.time_and_memory()
    options['max_depth'] = _NUM_OPS_TO_PRINT
    options['order_by'] = 'accelerator_micros'
    profiler.profile_operations(options)



def _is_mkl_flag_absent(mkl_flag):
    return not (absl_flags.FLAGS.is_parsed() and mkl_flag in absl_flags.FLAGS
                and absl_flags.FLAGS[mkl_flag].present)


def _print_os_env_ignored_warning(mkl_flag, flag_default_val, os_env_var):
    tf.logging.warn(
        ('OS ENV variable %s=%s is ignored and script default: '
         '%s is used. Use --%s to override.') %
        (os_env_var, os.environ[os_env_var], flag_default_val, mkl_flag))


def setup(params):
    """Sets up the environment that BenchmarkCNN should run in.

    Args:
      params: Params tuple, typically created by make_params or
              make_params_from_flags.
    Returns:
      A potentially modified params.
    Raises:
      ValueError: invalid parames combinations.
    """
    print('setting up params')
    # Create a dummy session to initialize TF global variables using the input
    # params. See b/115772076#comment6 for more details.
    with tf.Session(config=create_config_proto(params)) as sess:
        del sess

    if params.batchnorm_persistent:
        os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    else:
        os.environ.pop('TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT', None)
    if params.winograd_nonfused:
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    else:
        os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)
    if params.autotune_threshold:
        os.environ['TF_AUTOTUNE_THRESHOLD'] = str(params.autotune_threshold)
    os.environ['TF_SYNC_ON_FINISH'] = str(int(params.sync_on_finish))
    argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Sets environment variables for MKL
    # If OS ENV vars are overridden by script defaults, a warning msg is printed.
    if params.mkl:
        mkl_flags = ['kmp_blocktime', 'kmp_settings', 'kmp_affinity',
                     'num_intra_threads']
        for mkl_flag in mkl_flags:
            os_env_var = mkl_flag.upper()
            if mkl_flag == 'num_intra_threads':
                os_env_var = 'OMP_NUM_THREADS'
            flag_val = str(getattr(params, mkl_flag))
            if _is_mkl_flag_absent(mkl_flag) and os_env_var in os.environ:
                _print_os_env_ignored_warning(mkl_flag, flag_val, os_env_var)
            os.environ[os_env_var] = flag_val
            if mkl_flag == 'num_intra_threads' and not params.num_intra_threads:
                os.environ.pop(os_env_var, None)

    # Sets GPU thread settings
    if params.device.lower() == 'gpu':
        params = params._replace(gpu_thread_mode=params.gpu_thread_mode.lower())
        if params.gpu_thread_mode not in ['global', 'gpu_shared', 'gpu_private']:
            raise ValueError('Invalid gpu_thread_mode: %s' % params.gpu_thread_mode)
        os.environ['TF_GPU_THREAD_MODE'] = params.gpu_thread_mode

        if params.per_gpu_thread_count and params.gpu_thread_mode == 'global':
            raise ValueError(
                'Invalid per_gpu_thread_count with gpu_thread_mode=global: %s' %
                params.per_gpu_thread_count)
        # Default to two threads. One for the device compute and the other for
        # memory copies.
        per_gpu_thread_count = params.per_gpu_thread_count or 2
        total_gpu_thread_count = per_gpu_thread_count * params.num_gpus

        if params.gpu_thread_mode == 'gpu_private':
            os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
        elif params.gpu_thread_mode == 'gpu_shared':
            os.environ['TF_GPU_THREAD_COUNT'] = str(total_gpu_thread_count)

        cpu_count = multiprocessing.cpu_count()
        if not params.num_inter_threads and params.gpu_thread_mode in [
            'gpu_private', 'gpu_shared'
        ]:
            main_thread_count = max(cpu_count - total_gpu_thread_count, 1)
            params = params._replace(num_inter_threads=main_thread_count)

        if (params.datasets_use_prefetch and
                    params.datasets_num_private_threads is None):
            # From the total cpu thread count, subtract the total_gpu_thread_count,
            # and then 2 threads per GPU device for event monitoring and sending /
            # receiving tensors
            num_monitoring_threads = 2 * params.num_gpus
            num_private_threads = max(
                cpu_count - total_gpu_thread_count - num_monitoring_threads, 1)
            params = params._replace(datasets_num_private_threads=num_private_threads)

    if params.variable_update == 'horovod':
        import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
        hvd.init()

    platforms_util.initialize(params, create_config_proto(params))

    return params


def maybe_compile(computation, params):
    if params and params.xla_compile:
        return xla.compile(computation)
    else:
        return computation()


