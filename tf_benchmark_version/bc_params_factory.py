from bc_helpers import make_params_from_flags
import constants
from collections import namedtuple
import numpy as np

_DEFAULT_NUM_BATCHES = 100

CIFAR10_DATASET_DIR = '/home/dingxiaohan/slim_cifar10_tfrecords/'

ParamGroup = namedtuple('Params', ['lr_boundaries', 'deps', 'depth_multiplier',     'model', 'eval', 'eval_interval_secs', 'forward_only', 'freeze_when_forward_only', 'print_training_accuracy', 'batch_size', 'batch_group_size', 'num_batches', 'num_epochs', 'num_warmup_batches', 'autotune_threshold', 'num_gpus', 'gpu_indices', 'display_every', 'data_dir', 'data_name', 'resize_method', 'distortions', 'use_datasets', 'input_preprocessor', 'gpu_thread_mode', 'per_gpu_thread_count', 'hierarchical_copy', 'network_topology', 'gradient_repacking', 'compact_gradient_transfer', 'variable_consistency', 'datasets_repeat_cached_sample', 'local_parameter_device', 'device', 'data_format', 'num_intra_threads', 'num_inter_threads', 'trace_file', 'use_chrome_trace_format', 'tfprof_file', 'graph_file', 'partitioned_graph_file_prefix', 'optimizer', 'init_learning_rate', 'piecewise_learning_rate_schedule', 'num_epochs_per_decay', 'learning_rate_decay_factor', 'num_learning_rate_warmup_epochs', 'minimum_learning_rate', 'momentum', 'rmsprop_decay', 'rmsprop_momentum', 'rmsprop_epsilon', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'gradient_clip', 'weight_decay', 'gpu_memory_frac_for_testing', 'use_unified_memory', 'use_tf_layers', 'tf_random_seed', 'debugger', 'use_python32_barrier', 'datasets_use_prefetch', 'datasets_prefetch_buffer_size', 'datasets_num_private_threads', 'datasets_use_caching', 'datasets_parallel_interleave_cycle_length', 'datasets_sloppy_parallel_interleave', 'use_multi_device_iterator', 'multi_device_iterator_max_buffer_size', 'winograd_nonfused', 'batchnorm_persistent', 'sync_on_finish', 'staged_vars', 'force_gpu_compatible', 'allow_growth', 'xla', 'xla_compile', 'fuse_decode_and_crop', 'distort_color_in_yiq', 'enable_optimizations', 'rewriter_config', 'loss_type_to_report', 'single_l2_loss_op', 'use_resource_vars', 'mkl', 'kmp_blocktime', 'kmp_affinity', 'kmp_settings', 'use_fp16', 'fp16_loss_scale', 'fp16_vars', 'fp16_enable_auto_loss_scale', 'fp16_inc_loss_scale_every_n', 'variable_update', 'all_reduce_spec', 'agg_small_grads_max_bytes', 'agg_small_grads_max_group', 'allreduce_merge_scope', 'job_name', 'ps_hosts', 'worker_hosts', 'controller_host', 'task_index', 'server_protocol', 'cross_replica_sync', 'horovod_device', 'summary_verbosity', 'save_summaries_steps', 'save_model_secs', 'save_model_steps', 'max_ckpts_to_keep', 'train_dir', 'eval_dir', 'backbone_model_path', 'trt_mode', 'trt_max_workspace_size_bytes', 'benchmark_log_dir', 'benchmark_test_id',
                                   'label_smoothing',
                                   'adjust_dropout_rate_thresh'])

MyParam = namedtuple('MyParams', ['force_subset', 'init_hdf5', 'save_hdf5', 'num_steps_per_hdf5', 'show_variables', 'eval_log_file', 'just_compile',
                                  'save_mvav', 'load_ckpt', 'init_global_step', 'auto_continue', 'frequently_save_interval', 'frequently_save_last_epochs',
                                  'should_write_graph', 'need_record_internal_outputs', 'apply_l2_on_vector_params',
                                  'use_dense_layer', 'double_gradient_keywords', 'input_rotation'])

BDSParam = namedtuple('BDSParams', ['metric_type', 'train_batches_per_half', 'inc_ratio_limit', 'bds_log_file', 'save_per_moves', 'target_layers',
                                    'bds_start_step', 'flops_pruned_calc_fn', 'flops_calc_baseline_deps', 'terminate_at_pruned_flops', 'use_single_dropout'])


def decide_default_preprocess_name(model_name):
    return 'std'

def decide_save_model_steps(model_name, dataset):
    if dataset == 'imagenet':
        print('save original tf ckpt every 50000 steps for imagenet')
        return 50000
    else:
        print('save original tf ckpt every 100000 steps for {} on {}'.format(model_name, dataset))
        return 100000

def decide_default_dataset_name(model_name):
    return 'cifar10'


def decide_dataset_dir(dataset_name, model_name):
    return CIFAR10_DATASET_DIR


def decide_default_eval_batch_size(model_name):
    return 500

def decide_default_eval_num_batches(model_name):
    return 20


class Temp(object):

    def __init__(self):
        super(Temp, self).__init__()

def _get_original_default_params():
    # de_type = namedtuple('Params', [])
    de = Temp()
    de.model = 'trivial'
    de.eval = False
    de.eval_interval_secs = 0
    de.forward_only = False
    de.freeze_when_forward_only = False
    de.print_training_accuracy = False
    de.batch_size = 0                       #   per device
    de.batch_group_size = 1
    de.num_batches = None
    de.num_epochs = None                    #   excluding warmup, this and num_batches cannot both be specified
    de.num_warmup_batches = None
    de.autotune_threshold = None
    de.num_gpus = 1
    de.gpu_indices = ''
    de.display_every = 10
    de.data_dir = None
    de.data_name = None
    de.resize_method = 'bilinear'
    de.distortions = True
    de.use_datasets = True
    de.input_preprocessor = 'default'
    de.gpu_thread_mode = 'gpu_private'
    de.per_gpu_thread_count = 0
    de.hierarchical_copy = False
    de.network_topology = constants.NetworkTopology.DGX1    #   originally an enum, I think this is useless
    de.gradient_repacking = 0
    de.compact_gradient_transfer = True
    de.variable_consistency = 'strong'          #   originally an enum, interesting
    de.datasets_repeat_cached_sample = False
    de.local_parameter_device = 'gpu'
    de.device = 'gpu'
    de.data_format = 'NCHW'             #Data layout to use: NHWC (TF native) or NCHW (cuDNN native, requires GPU).
    de.num_intra_threads = None
    de.num_inter_threads = 0
    de.trace_file = ''
    de.use_chrome_trace_format = True

    de.tfprof_file = None
    de.graph_file = None
    de.partitioned_graph_file_prefix = None
    de.optimizer = 'sgd'                #   ('momentum', 'sgd', 'rmsprop', 'adam')

    #   lr stuff
    de.init_learning_rate = None
    de.piecewise_learning_rate_schedule = None
    de.num_epochs_per_decay = 0
    de.learning_rate_decay_factor = 0
    de.num_learning_rate_warmup_epochs = 0
    de.minimum_learning_rate = 0
    de.momentum = 0.9
    de.rmsprop_decay = 0.9
    de.rmsprop_momentum = 0.9
    de.rmsprop_epsilon = 1.0
    de.adam_beta1 = 0.9
    de.adam_beta2 = 0.999
    de.adam_epsilon = 1e-8


    de.gradient_clip = None
    de.weight_decay = 0
    de.gpu_memory_frac_for_testing = 0

    de.use_unified_memory = False
    de.use_tf_layers = True
    de.tf_random_seed = 1234            #   Useful for debugging NaNs, as this can be set to various values to see if the NaNs depend on the seed.
    de.debugger = None                  #   interesting
    de.use_python32_barrier = False

    de.datasets_use_prefetch = True
    de.datasets_prefetch_buffer_size = 1
    de.datasets_num_private_threads = None
    de.datasets_use_caching = False    # iteresting! Cache the compressed input data in memory. This improves the data input performance, at the cost of additional memory.' TODO
    de.datasets_parallel_interleave_cycle_length = None
    de.datasets_sloppy_parallel_interleave = False

    de.use_multi_device_iterator = True # If true, we use the MultiDeviceIterator for prefetching, which deterministically prefetches the data onto the various GPUs
    de.multi_device_iterator_max_buffer_size = 1

    #   do not know what the following things are, better ignore them
    de.winograd_nonfused = True
    de.batchnorm_persistent = True
    de.sync_on_finish = False           #   interesting
    de.staged_vars = False              #   this seems interesting
    de.force_gpu_compatible = False

    de.allow_growth = None              #   should be True in my codes

    de.xla = False
    de.xla_compile = False

    #   for preprocessing
    de.fuse_decode_and_crop = True
    de.distort_color_in_yiq = True

    de.enable_optimizations = True      #   do not know what this is
    de.rewriter_config = None


    # ('base_loss', 'total_loss'),
    # 'Which type of loss to output and to write summaries for. '
    # 'The total loss includes L2 loss while the base loss does '
    # 'not. Note that the total loss is always used while '
    # 'computing gradients during training if weight_decay > 0, '
    # 'but explicitly computing the total loss, instead of just '
    # 'computing its gradients, can have a performance impact.'
    de.loss_type_to_report = 'total_loss'
    de.single_l2_loss_op = False

    de.use_resource_vars = False
    de.mkl = False
    de.kmp_blocktime = 0
    de.kmp_affinity = 'granularity=fine,verbose,compact,1,0'
    de.kmp_settings = 1


    de.use_fp16 = False
    de.fp16_loss_scale = None
    de.fp16_vars = False
    de.fp16_enable_auto_loss_scale = False
    de.fp16_inc_loss_scale_every_n = 1000

    de.variable_update = 'parameter_server'     #   but for resnet-50 replicated is recommended

    #   do not touch these
    de.all_reduce_spec = None
    de.agg_small_grads_max_bytes = 0
    de.agg_small_grads_max_group = 10
    de.allreduce_merge_scope = 1


    #   for distributed training
    de.job_name = ''
    de.ps_hosts = ''
    de.worker_hosts = ''
    de.controller_host = None
    de.task_index = 0
    de.server_protocol = 'grpc'
    de.cross_replica_sync = True
    de.horovod_device = ''


    #   for summary and S/L ckpts
    de.summary_verbosity = 0
    de.save_summaries_steps= 0
    de.save_model_secs = 0
    de.save_model_steps = None
    de.max_ckpts_to_keep = 5
    de.train_dir = None
    de.eval_dir = '/tmp/tf_cnn_benchmarks/eval'
    de.backbone_model_path = None
    de.trt_mode = ''
    de.trt_max_workspace_size_bytes = 4 << 30

    de.benchmark_log_dir = None
    de.benchmark_test_id = None

    #   shawn
    de.deps = None
    de.depth_multiplier = 1
    de.lr_boundaries = None
    de.label_smoothing = -1.0
    de.adjust_dropout_rate_thresh = None

    return ParamGroup(**de.__dict__)


#   python tf_cnn_benchmarks.py --summary_verbosity=3 --save_summaries_steps=100 --save_model_secs=3600 --train_dir=lct_resnet18_gpu4_batch256_train --allow_growth --max_ckpts_to_keep=10

#   take all the default params and modify the needed ones
def default_params_for_train(model_name, train_dir, batch_size, optimizer_name, num_gpus, weight_decay,
                             max_epochs, lr_warmup_epochs=0, init_lr=None, num_epochs_per_decay=None, lr_decay_factor=None, lr_epoch_boundaries=None, lr_values=None,
                             dataset_name=None, use_default_lr=False, preprocessor=None, use_distortions=True, data_format='NHWC', deps=None,
                             depth_multiplier=1, save_model_steps=None, save_summaries_steps=100, summary_verbosity=1, gradient_clip=None, label_smoothing=-1.0,
                             adjust_dropout_rate_thresh=None, momentum=0.9):
    defaults = _get_original_default_params()

    # if 'mobi' in model_name:
    #     assert data_format == 'NHWC'
    #
    # if model_name == 'mobilenet':
    #     assert data_format == 'NHWC'        #   downloaded from the slim repo


    # assert optimizer_name != 'sgd'

    defaults = defaults._replace(model=model_name)
    defaults = defaults._replace(train_dir=train_dir)
    defaults = defaults._replace(optimizer=optimizer_name)
    defaults = defaults._replace(batch_size=batch_size)
    defaults = defaults._replace(num_gpus=num_gpus)
    defaults = defaults._replace(num_epochs=max_epochs)
    defaults = defaults._replace(weight_decay=weight_decay)
    defaults = defaults._replace(input_preprocessor=preprocessor or decide_default_preprocess_name(model_name))
    defaults = defaults._replace(distortions=use_distortions)
    defaults = defaults._replace(data_format=data_format)
    defaults = defaults._replace(deps=deps)
    defaults = defaults._replace(depth_multiplier=depth_multiplier)

    defaults = defaults._replace(variable_update='replicated')
    defaults = defaults._replace(summary_verbosity=summary_verbosity)
    defaults = defaults._replace(save_summaries_steps=100)

    defaults = defaults._replace(allow_growth=True)
    defaults = defaults._replace(max_ckpts_to_keep=999999)
    defaults = defaults._replace(loss_type_to_report='base_loss')

    dataset = dataset_name or decide_default_dataset_name(model_name)
    defaults = defaults._replace(data_name=dataset)

    defaults = defaults._replace(save_model_steps=save_model_steps or decide_save_model_steps(model_name, dataset))

    if defaults.data_name in ['ch', 'cifar10']:
        assert use_distortions

    defaults = defaults._replace(data_dir=decide_dataset_dir(defaults.data_name, model_name=model_name))

    if not use_default_lr:
        if init_lr is None:
            assert lr_values is not None
            schedule_str = str(lr_values[0])
            for i in range(len(lr_epoch_boundaries)):
                schedule_str += ';{};{}'.format(lr_epoch_boundaries[i], lr_values[i + 1])
            defaults = defaults._replace(piecewise_learning_rate_schedule=schedule_str)
        else:
            assert lr_values is None
            defaults = defaults._replace(num_epochs_per_decay=num_epochs_per_decay)
            defaults = defaults._replace(learning_rate_decay_factor=lr_decay_factor)
            defaults = defaults._replace(init_learning_rate=init_lr)

        defaults = defaults._replace(num_learning_rate_warmup_epochs=lr_warmup_epochs)
    else:
        assert lr_values is None and lr_epoch_boundaries is None and init_lr is None

    defaults = defaults._replace(gradient_clip=gradient_clip)
    defaults = defaults._replace(label_smoothing=label_smoothing)
    defaults = defaults._replace(adjust_dropout_rate_thresh=adjust_dropout_rate_thresh)
    defaults = defaults._replace(momentum=momentum)
    if momentum != 0.9:
        print('----------------------------------------')
        print('NOTE that momentum = ', momentum)
        print('----------------------------------------')

    return defaults


#   take all the default params and modify the needed ones
def default_params_for_eval(model_name, eval_dir=None, dataset_name=None, preprocessor=None, data_format='NHWC', deps=None, depth_multiplier=1, num_batches=None, batch_size=None):

    # if 'mobi' in model_name:
    #     assert data_format == 'NHWC'
    #
    # # if model_name in ['alexnet']:
    # #     assert data_format == 'NCHW'  # trained in NCHW format from scratch
    # if model_name == 'mobilenet':
    #     assert data_format == 'NHWC'  # downloaded from the slim repo

    defaults = _get_original_default_params()

    defaults = defaults._replace(model=model_name)
    defaults = defaults._replace(batch_size=batch_size or decide_default_eval_batch_size(model_name))
    defaults = defaults._replace(num_batches=num_batches or decide_default_eval_num_batches(model_name))
    defaults = defaults._replace(deps=deps)
    defaults = defaults._replace(depth_multiplier=depth_multiplier)

    defaults = defaults._replace(num_gpus=1)
    defaults = defaults._replace(eval=True)
    defaults = defaults._replace(eval_dir=None)

    defaults = defaults._replace(data_format=data_format)
    defaults = defaults._replace(variable_update='replicated')
    defaults = defaults._replace(summary_verbosity=0)
    defaults = defaults._replace(allow_growth=True)

    preprocess_name = preprocessor or decide_default_preprocess_name(model_name)
    dataset_name = dataset_name or decide_default_dataset_name(model_name)
    # if dataset_name in ['cifar10', 'ch']:
    #     dataset_name += '_record'
    #     preprocess_name += '_record'
    defaults = defaults._replace(data_name=dataset_name)
    defaults = defaults._replace(input_preprocessor=preprocess_name)

    defaults = defaults._replace(data_dir=decide_dataset_dir(defaults.data_name, model_name=model_name))

    return defaults


def default_my_params(init_hdf5=None, save_hdf5=None, num_steps_per_hdf5=10000, show_variables=False, force_subset=None, eval_log_file=None, just_compile=False,
                      save_mvav=False, load_ckpt=None, init_global_step=None, auto_continue=False, frequently_save_interval=None, frequently_save_last_epochs=None,
                      should_write_graph=False,
                      need_record_internal_outputs=False,
                      apply_l2_on_vector_params=False,
                      use_dense_layer=False,
                      double_gradient_keywords=('bias', 'beta'),
                      input_rotation=0):
    result = MyParam(force_subset=force_subset, init_hdf5=init_hdf5, save_hdf5=save_hdf5, num_steps_per_hdf5=num_steps_per_hdf5, show_variables=show_variables, eval_log_file=eval_log_file, just_compile=just_compile,
        save_mvav=save_mvav, load_ckpt=load_ckpt, init_global_step=init_global_step, auto_continue=auto_continue, frequently_save_interval=frequently_save_interval, frequently_save_last_epochs=frequently_save_last_epochs,
        should_write_graph=should_write_graph, need_record_internal_outputs=need_record_internal_outputs, apply_l2_on_vector_params=apply_l2_on_vector_params,
        use_dense_layer=use_dense_layer, double_gradient_keywords=double_gradient_keywords, input_rotation=input_rotation)
    return result



def get_bds_params(metric_type='euc', train_batches_per_half=None, inc_ratio_limit=None, bds_log_file=None, save_per_moves=None, target_layers=None,
                   bds_start_step=None,flops_pruned_calc_fn=None,flops_calc_baseline_deps=None,terminate_at_pruned_flops=None, use_single_dropout=False):
    return BDSParam(metric_type=metric_type, train_batches_per_half=train_batches_per_half, inc_ratio_limit=inc_ratio_limit,
        bds_log_file=bds_log_file, save_per_moves=save_per_moves, target_layers=list(target_layers), bds_start_step=bds_start_step,
        flops_pruned_calc_fn=flops_pruned_calc_fn,flops_calc_baseline_deps=flops_calc_baseline_deps,terminate_at_pruned_flops=terminate_at_pruned_flops, use_single_dropout=use_single_dropout)