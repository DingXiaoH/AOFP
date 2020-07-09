from bc import BenchmarkCNN
from bc_params_factory import *
import sys
from bc_constants import VGG_ORIGIN_DEPS
from tf_utils import extract_deps_from_weights_file
from aofp.bds_prune_base import bds_prune_base_pipeline
from util_scripts.compression_calculat import calculate_vfs_flops

BASE_WEIGHTS = 'vc_scratch_savedweights.hdf5'

def bds_prune_vc(try_arg,batches_per_half,metric_limit,
                      max_epochs,lr_warmup_epochs,lr_epoch_boundaries,lr_values,
                      num_gpus,
                      ft_max_epochs, ft_lr_warmup_epochs, ft_lr_epoch_boundaries, ft_lr_values,
                      save_per_moves=10,
                 skip_train=False,
                        prune_from=None,
                        ft_from_other_pruned_weights=None,
                       ft_load_ckpt=None,
                       ft_init_step = 0,

                       train_preprocess='std',
                       ft_preprocess='std',
                       start_weights=BASE_WEIGHTS,
                        terminate_flops= 1.0,
                 use_single_dropout=False
                      ):

    bds_prune_base_pipeline(model_name='vc',
                            fc_layer_idxes=[13],
                            subsequent_strategy='simple',
                            target_layers=range(0, 13),
                            try_arg=try_arg,
                            init_hdf5=start_weights,
                            batches_per_half=batches_per_half,
                            metric_limit=metric_limit,
                            batch_size=64,
                            weight_decay=1e-4,
                            max_epochs=max_epochs, lr_warmup_epochs=lr_warmup_epochs, lr_epoch_boundaries=lr_epoch_boundaries, lr_values=lr_values,
                            use_distortions=True,
                            data_format='NHWC',
                            deps=extract_deps_from_weights_file(start_weights),
                            num_steps_per_hdf5=20000,
                            num_gpus=num_gpus,
                            save_per_moves=save_per_moves,
                            ft_max_epochs=ft_max_epochs, ft_lr_warmup_epochs=ft_lr_warmup_epochs, ft_lr_epoch_boundaries=ft_lr_epoch_boundaries, ft_lr_values=ft_lr_values,
                            skip_train=skip_train,
                            prune_from=prune_from,
                            flops_pruned_calc_fn=calculate_vfs_flops,
                            flops_calc_baseline_deps=np.array(VGG_ORIGIN_DEPS),
        ft_from_other_pruned_weights=ft_from_other_pruned_weights,
        ft_load_ckpt=ft_load_ckpt,
        ft_init_step=ft_init_step,

        train_preprocess=train_preprocess,
        ft_preprocess=ft_preprocess,

        apply_l2_on_vecs=False,
        use_dense_layer=True,
        terminate_at_pruned_flops=terminate_flops,
    use_single_dropout=use_single_dropout)

def bds_prune_vc_multi_itrs(try_arg_base, itr_schedule, batches_per_half,metric_limit,
                      max_epochs,lr_warmup_epochs,lr_epoch_boundaries,lr_values,
                      num_gpus,
                      ft_max_epochs, ft_lr_warmup_epochs, ft_lr_epoch_boundaries, ft_lr_values,
                        restore_itr=0,
                            first_itr_start_weights=BASE_WEIGHTS,
                            use_single_dropout=False):
    for itr_idx, terminate_flops in enumerate(itr_schedule):
        if itr_idx < restore_itr:
            continue
        try_arg = try_arg_base + '_itr{}'.format(itr_idx)
        if itr_idx == 0:
            start_weights = first_itr_start_weights
        else:
            start_weights = 'bds_exps/{}_{}_ftedweights.hdf5'.format('vc', try_arg_base + '_itr{}'.format(itr_idx - 1))
        bds_prune_vc(try_arg=try_arg, batches_per_half=batches_per_half, metric_limit=metric_limit,
            max_epochs=max_epochs, lr_warmup_epochs=lr_warmup_epochs, lr_epoch_boundaries=lr_epoch_boundaries, lr_values=lr_values, num_gpus=num_gpus,
            ft_max_epochs=ft_max_epochs, ft_lr_warmup_epochs=ft_lr_warmup_epochs, ft_lr_epoch_boundaries=ft_lr_epoch_boundaries, ft_lr_values=ft_lr_values,
            start_weights=start_weights, terminate_flops=terminate_flops, use_single_dropout=use_single_dropout)

if __name__ == '__main__':

    command = sys.argv[1]
    assert command in ['train', 'eval', 'prune']

    if command == 'train':
        params = default_params_for_train('vc', train_dir='vc_scratch_train',
            batch_size=64, optimizer_name='momentum', num_gpus=1, weight_decay=1e-4,
            max_epochs=600, lr_warmup_epochs=0, lr_epoch_boundaries=[200, 400], lr_values=[5e-2, 5e-3, 5e-4],
            dataset_name='cifar10', use_default_lr=False, preprocessor='std', data_format='NHWC', deps=VGG_ORIGIN_DEPS,
            save_model_steps=100000)
        my_params = default_my_params(init_hdf5=None, save_hdf5=BASE_WEIGHTS,
            apply_l2_on_vector_params=False, use_dense_layer=True)
        bc = BenchmarkCNN(params=params, my_params=my_params)
        bc.print_info()
        bc.run()
    elif command == 'eval':
        target = sys.argv[2]
        params = default_params_for_eval('vc', batch_size=200, num_batches=50, dataset_name='cifar10', preprocessor='std', data_format='NHWC',
            deps=extract_deps_from_weights_file(target))
        my_params = default_my_params(init_hdf5=target, use_dense_layer=True)
        bc = BenchmarkCNN(params=params, my_params=my_params)
        bc.print_info()
        bc.run()
    elif command == 'prune':
        bds_prune_vc_multi_itrs(try_arg_base='reproduce', itr_schedule=[0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9],
            batches_per_half=20000, metric_limit=0.01,
            max_epochs=9999, lr_warmup_epochs=0, lr_epoch_boundaries=[9999], lr_values=[1e-3, 1e-3],
            num_gpus=1,
            ft_max_epochs=600, ft_lr_warmup_epochs=10, ft_lr_epoch_boundaries=[200, 400],
            ft_lr_values=[5e-2, 5e-3, 5e-4])
    else:
        assert False

