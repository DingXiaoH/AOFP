from bc_params_factory import *
from aofp.bds_bc import BDSBenchmark
from aofp.bds_utils import extract_deps_by_mask_value_from_hdf5, tfm_prune_filters_and_save_by_mask
from bc import BenchmarkCNN
import os
from tf_utils import extract_deps_from_weights_file

BDS_OVERALL_LOG_FILE = 'aofp_overall_logs.txt'


def bds_prune_base_pipeline(model_name,
                            fc_layer_idxes,
                            subsequent_strategy,
                            target_layers,
                            try_arg,
                            init_hdf5,
                            batches_per_half,
                            metric_limit,
                            batch_size,
                            weight_decay,
                            max_epochs,lr_warmup_epochs,lr_epoch_boundaries,lr_values,
                            use_distortions,
                            data_format,
                            deps,
                            num_steps_per_hdf5,
                            num_gpus,
                            save_per_moves,
ft_max_epochs,ft_lr_warmup_epochs,ft_lr_epoch_boundaries,ft_lr_values,

                            skip_train,
                            prune_from=None,
                            bds_start_step=2000,
                            restore_train_step=0,
                            auto_continue_st=False,
                            flops_pruned_calc_fn=None,
                            flops_calc_baseline_deps=None,

                            ft_from_other_pruned_weights=None,
                            ft_load_ckpt=None,
                            ft_init_step=0,

                            train_preprocess=None,
                            ft_preprocess=None,
apply_l2_on_vecs = False,
                            use_dense_layer=False,
                            terminate_at_pruned_flops=1.0,
                            bds_batch_size=None,
                            use_single_dropout=False
                            ):


    tuned_hdf5 = ft_from_other_pruned_weights or 'bds_exps/{}_{}_tunedweights.hdf5'.format(model_name, try_arg)

    if ft_from_other_pruned_weights is None:
        if not skip_train:
            assert not auto_continue_st
            train_dir = 'bds_exps/{}_{}_train'.format(model_name, try_arg)
            tune_params = default_params_for_train(model_name=model_name,
                train_dir=train_dir, batch_size=bds_batch_size or batch_size,
                optimizer_name='momentum', num_gpus=num_gpus,
                weight_decay=weight_decay,
                max_epochs=max_epochs, lr_warmup_epochs=lr_warmup_epochs, lr_epoch_boundaries=lr_epoch_boundaries,
                lr_values=lr_values,
                use_default_lr=False,
                use_distortions=use_distortions,
                data_format=data_format,
                deps=deps, preprocessor=train_preprocess)
            if restore_train_step > 0:
                restore_global_step = restore_train_step
                restore_weights = os.path.join(train_dir, 'ckpt_step_{}.hdf5'.format(restore_train_step))
            else:
                restore_global_step = 0
                restore_weights = init_hdf5
            tune_my_params = default_my_params(init_hdf5=restore_weights, save_hdf5=tuned_hdf5,
                num_steps_per_hdf5=num_steps_per_hdf5, show_variables=False, auto_continue=False,
                should_write_graph=True,
                save_mvav=False, init_global_step=restore_global_step, apply_l2_on_vector_params=apply_l2_on_vecs, use_dense_layer=use_dense_layer)
            tune_bds_params = get_bds_params(train_batches_per_half=batches_per_half, inc_ratio_limit=metric_limit,
                bds_log_file='bds_exps/{}_{}_bdslog.txt'.format(model_name, try_arg),
                save_per_moves=save_per_moves, target_layers=target_layers, bds_start_step=bds_start_step,
                flops_pruned_calc_fn=flops_pruned_calc_fn,flops_calc_baseline_deps=flops_calc_baseline_deps,
                terminate_at_pruned_flops=terminate_at_pruned_flops, use_single_dropout=use_single_dropout
            )

            bds_bc = BDSBenchmark(params=tune_params, my_params=tune_my_params, bds_params=tune_bds_params)
            bds_bc.run()
            del bds_bc

        pruned_weights = 'bds_exps/{}_{}_prunedweights.hdf5'.format(model_name, try_arg)
        weights_to_prune = prune_from or tuned_hdf5
        if not auto_continue_st:
            #   test with the masks and report the acc
            first_eval_params = default_params_for_eval(model_name=model_name, data_format=data_format, deps=deps)
            first_eval_my_params = default_my_params(init_hdf5=weights_to_prune, eval_log_file=BDS_OVERALL_LOG_FILE,
                just_compile=True, use_dense_layer=use_dense_layer)
            first_eval_bds_params = get_bds_params(target_layers=target_layers, bds_start_step=bds_start_step)
            first_eval_bc = BDSBenchmark(params=first_eval_params, my_params=first_eval_my_params,
                bds_params=first_eval_bds_params)
            first_eval_bc.run()
            first_eval_bc.load_weights_from_hdf5(weights_to_prune)
            result_deps = first_eval_bc.get_deps_by_mask_value()
            first_eval_bc.simple_eval(
                eval_record_comment=' after tuning, the masks suggest deps={}'.format(list(result_deps)),
                other_log_file=BDS_OVERALL_LOG_FILE)

            #   prune and test
            tfm_prune_filters_and_save_by_mask(first_eval_bc, layer_to_masks=first_eval_bc.layer_to_mask_value,
                save_file=pruned_weights, fc_layer_idxes=fc_layer_idxes, subsequent_strategy=subsequent_strategy,
                result_deps=result_deps, data_format=data_format)
            del first_eval_bc
            second_eval_params = default_params_for_eval(model_name=model_name, data_format=data_format,
                deps=result_deps)
            second_eval_my_params = default_my_params(init_hdf5=pruned_weights, eval_log_file=BDS_OVERALL_LOG_FILE, use_dense_layer=use_dense_layer)
            second_bc = BenchmarkCNN(params=second_eval_params, my_params=second_eval_my_params)
            second_bc.run()
            del second_bc
        else:
            result_deps = extract_deps_by_mask_value_from_hdf5(tuned_hdf5)
    else:
        pruned_weights = ft_from_other_pruned_weights
        result_deps = extract_deps_from_weights_file(pruned_weights)

    #   finetune
    print('finetune from {}'.format(pruned_weights))
    if ft_max_epochs > 0:
        fted_weights = 'bds_exps/{}_{}_ftedweights.hdf5'.format(model_name, try_arg)
        ft_params = default_params_for_train(model_name=model_name,
            train_dir='bds_exps/{}_{}_ft'.format(model_name, try_arg), batch_size=batch_size,
            optimizer_name='momentum', num_gpus=num_gpus,
            weight_decay=weight_decay,
            max_epochs=ft_max_epochs, lr_warmup_epochs=ft_lr_warmup_epochs, lr_epoch_boundaries=ft_lr_epoch_boundaries,
            lr_values=ft_lr_values,
            use_default_lr=False,
            use_distortions=use_distortions,
            data_format=data_format,
            deps=result_deps,
            preprocessor=ft_preprocess)
        ft_my_params = default_my_params(init_hdf5=pruned_weights, save_hdf5=fted_weights,
            num_steps_per_hdf5=num_steps_per_hdf5, show_variables=False, auto_continue=True, should_write_graph=True,
            save_mvav=False, load_ckpt=ft_load_ckpt, init_global_step=ft_init_step, apply_l2_on_vector_params=apply_l2_on_vecs, use_dense_layer=use_dense_layer,

            frequently_save_last_epochs=50 if model_name == 'vc' else None, frequently_save_interval=2000 if model_name == 'vc' else None)
        ft_bc = BenchmarkCNN(params=ft_params, my_params=ft_my_params)
        ft_bc.run()
        del ft_bc

        #   test at last
        last_eval_params = default_params_for_eval(model_name=model_name, data_format=data_format, deps=result_deps)
        last_eval_my_params = default_my_params(init_hdf5=fted_weights, eval_log_file=BDS_OVERALL_LOG_FILE, use_dense_layer=use_dense_layer)
        last_bc = BenchmarkCNN(params=last_eval_params, my_params=last_eval_my_params)
        last_bc.run()
        del last_bc




