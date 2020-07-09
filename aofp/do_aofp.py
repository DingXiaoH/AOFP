from constants import *
from base_config import get_baseconfig_by_epoch
from model_map import get_dataset_name_by_model_name
import argparse
from ndp_test import general_test
import os
from aofp.aofp_builder import AOFPBuilder
from aofp.aofp_train import aofp_train_main
from aofp.flops_func import *
from ndp_train import train_main
from utils.misc import extract_deps_from_weights_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', default='vc')
    parser.add_argument('-c', '--conti_or_fs', default='fs')
    parser.add_argument(
        '--local_rank', default=0, type=int,
        help='process rank on node')

    start_arg = parser.parse_args()

    network_type = start_arg.arch
    conti_or_fs = start_arg.conti_or_fs
    assert conti_or_fs in ['continue', 'fs']
    auto_continue = conti_or_fs == 'continue'
    print('auto continue: ', auto_continue)

    if network_type == 'vc':
        weight_decay_strength = 1e-4
        batch_size = 64
        deps = VGG_ORIGIN_DEPS
        succ_strategy = {i:(i+1) for i in range(13)}
        init_hdf5 = 'vc_train/finish.hdf5'
        flops_func = calculate_vc_flops

        target_layers = list(range(13))
        iters_per_half = 1000
        thresh = 0.01
        warmup_iterations = 500
        flops_remain_target = 0.35

        # aofp_lrs = LRSchedule(base_lr=0.01, max_epochs=200, lr_epoch_boundaries=None, lr_decay_factor=None,
        #                  linear_final_lr=None, cosine_minimum=0)
        # aofp_lrs = LRSchedule(base_lr=0.001, max_epochs=1000, lr_epoch_boundaries=[1000], lr_decay_factor=0.1,
        #                  linear_final_lr=None, cosine_minimum=None)
        # finetune_lrs = LRSchedule(base_lr=0.002, max_epochs=600, lr_epoch_boundaries=None, lr_decay_factor=None,
        #                  linear_final_lr=None, cosine_minimum=0)

        aofp_lrs = LRSchedule(base_lr=0.001, max_epochs=600, lr_epoch_boundaries=[600], lr_decay_factor=0.1,
                              linear_final_lr=None, cosine_minimum=None)
        finetune_lrs = LRSchedule(base_lr=0.01, max_epochs=600, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)

        # finetune_lrs = LRSchedule(base_lr=0.04, max_epochs=400, lr_epoch_boundaries=None, lr_decay_factor=None,
        #                  linear_final_lr=None, cosine_minimum=0)

    elif network_type == 'src56':
        weight_decay_strength = 1e-4
        batch_size = 64
        deps = rc_origin_deps_flattened(9)
        succ_strategy = {i:(i+1) for i in rc_internal_layers(9)}
        init_hdf5 = 'src56_train/finish.hdf5'
        flops_func = calculate_rc56_flops

        target_layers = rc_internal_layers(9)
        iters_per_half = 1000
        thresh = 0.01
        warmup_iterations = 500
        flops_remain_target = 0.50

        # aofp_lrs = LRSchedule(base_lr=0.01, max_epochs=200, lr_epoch_boundaries=None, lr_decay_factor=None,
        #                  linear_final_lr=None, cosine_minimum=0)
        # finetune_lrs = LRSchedule(base_lr=0.04, max_epochs=400, lr_epoch_boundaries=None, lr_decay_factor=None,
        #                  linear_final_lr=None, cosine_minimum=0)
        aofp_lrs = LRSchedule(base_lr=0.001, max_epochs=600, lr_epoch_boundaries=[600], lr_decay_factor=0.1,
                              linear_final_lr=None, cosine_minimum=None)
        finetune_lrs = LRSchedule(base_lr=0.01, max_epochs=600, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)

        # aofp_lrs = LRSchedule(base_lr=0.001, max_epochs=1000, lr_epoch_boundaries=[1000], lr_decay_factor=0.1,
        #                       linear_final_lr=None, cosine_minimum=None)
        # finetune_lrs = LRSchedule(base_lr=0.05, max_epochs=600, lr_epoch_boundaries=None, lr_decay_factor=None,
        #                           linear_final_lr=None, cosine_minimum=0)
        #


    elif network_type == 'lenet5bn':
        weight_decay_strength = 5e-4
        batch_size = 256
        deps = [20, 50]
        succ_strategy = {i:(i+1) for i in [0, 1]}
        init_hdf5 = 'lenet5bn_lrsA.hdf5'
        flops_func = calculate_lenet5bn_flops


        # target_layers = [0, 1]
        # iters_per_half = 100
        # thresh = 0.2
        # warmup_iterations = 100
        # lrs = LRSchedule(base_lr=0.05, max_epochs=40, lr_epoch_boundaries=None, lr_decay_factor=None,
        #                  linear_final_lr=None, cosine_minimum=0)

        target_layers = [0, 1]
        iters_per_half = 400
        thresh = 0.05
        warmup_iterations = 100
        aofp_lrs = LRSchedule(base_lr=0.01, max_epochs=80, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        finetune_lrs = LRSchedule(base_lr=0.01, max_epochs=80, lr_epoch_boundaries=None, lr_decay_factor=None,
                              linear_final_lr=None, cosine_minimum=0)



    else:
        raise ValueError('...')

    log_dir = 'aofp_models/{}_train'.format(network_type)

    weight_decay_bias = 0
    warmup_factor = 0

    aofp_config = get_baseconfig_by_epoch(network_type=network_type,
                                     dataset_name=get_dataset_name_by_model_name(network_type), dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                     max_epochs=aofp_lrs.max_epochs, base_lr=aofp_lrs.base_lr,
                                     lr_epoch_boundaries=aofp_lrs.lr_epoch_boundaries, cosine_minimum=aofp_lrs.cosine_minimum,
                                     lr_decay_factor=aofp_lrs.lr_decay_factor,
                                     warmup_epochs=0, warmup_method='linear', warmup_factor=warmup_factor,
                                     ckpt_iter_period=40000, tb_iter_period=100, output_dir=log_dir,
                                     tb_dir=log_dir, save_weights=None, val_epoch_period=2, linear_final_lr=aofp_lrs.linear_final_lr,
                                     weight_decay_bias=weight_decay_bias, deps=deps)

    pruned_path = os.path.join(aofp_config.output_dir, 'finish_pruned.hdf5')
    if not os.path.exists(pruned_path):
        aofp_builder = AOFPBuilder(base_config=aofp_config, target_layers=target_layers,
                                 succ_strategy=succ_strategy, iters_per_half=iters_per_half, thresh=thresh)
        aofp_train_main(local_rank=start_arg.local_rank, target_layers=target_layers, succ_strategy=succ_strategy,
                        warmup_iterations=warmup_iterations, aofp_batches_per_half=iters_per_half,
                        cfg=aofp_config, show_variables=True, convbuilder=aofp_builder, init_hdf5=init_hdf5,
                        flops_func=flops_func, remain_flops_ratio=flops_remain_target)

    #   finetune
    pruned_deps = extract_deps_from_weights_file(pruned_path)
    finetune_config = get_baseconfig_by_epoch(network_type=network_type,
                                          dataset_name=get_dataset_name_by_model_name(network_type),
                                          dataset_subset='train',
                                          global_batch_size=batch_size, num_node=1,
                                          weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                          max_epochs=finetune_lrs.max_epochs, base_lr=finetune_lrs.base_lr,
                                          lr_epoch_boundaries=finetune_lrs.lr_epoch_boundaries,
                                          cosine_minimum=finetune_lrs.cosine_minimum,
                                          lr_decay_factor=finetune_lrs.lr_decay_factor,
                                          warmup_epochs=0, warmup_method='linear', warmup_factor=warmup_factor,
                                          ckpt_iter_period=40000, tb_iter_period=100, output_dir=log_dir,
                                          tb_dir=log_dir, save_weights=None, val_epoch_period=2,
                                          linear_final_lr=finetune_lrs.linear_final_lr,
                                          weight_decay_bias=weight_decay_bias, deps=pruned_deps)

    train_main(local_rank=start_arg.local_rank, cfg=finetune_config, show_variables=True,
               init_hdf5=pruned_path)



