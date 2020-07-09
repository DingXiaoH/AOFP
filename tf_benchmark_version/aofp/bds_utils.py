import numpy as np
from bc_constants import MASK_VALUE_KEYWORD
from tf_utils import read_hdf5, save_hdf5

def delete_or_keep(array, idxes, axis=None):
    if len(idxes) > 0:
        return np.delete(array, idxes, axis=axis)
    else:
        return array


def extract_deps_by_mask_value_from_hdf5(hdf5_path):
    hdf5_dict = read_hdf5(hdf5_path)
    layer_idx_to_dep = {}
    for n, v in hdf5_dict.items():
        if MASK_VALUE_KEYWORD in n:
            layer_idx_to_dep[int(n.replace(MASK_VALUE_KEYWORD, ''))] = int(np.sum(v > 0.99))
    deps_list = []
    for k in sorted(layer_idx_to_dep.keys()):
        deps_list.append(layer_idx_to_dep[k])
    return np.array(deps_list, dtype=np.int32)




def generate_base_mask_feed_dict(bds_bc, exp_layer_idx, cur_base_mask):
    base_mask_dict = {m: np.ones(m.get_shape()) for m in bds_bc.base_masks[0].values()}
    ds_mask_dict = {m: np.ones(m.get_shape()) for m in bds_bc.ds_masks[0].values()}
    base_mask_dict.update(ds_mask_dict)
    base_mask_dict[bds_bc.base_masks[0][exp_layer_idx]] = cur_base_mask
    return base_mask_dict

def generate_ds_mask_feed_dict(bds_bc, exp_layer_idx, cur_ds_mask):
    base_mask_dict = {m: np.ones(m.get_shape()) for m in bds_bc.base_masks.values()}
    ds_mask_dict = {m: np.ones(m.get_shape()) for m in bds_bc.ds_masks.values()}
    base_mask_dict.update(ds_mask_dict)
    base_mask_dict[bds_bc.ds_masks[exp_layer_idx]] = cur_ds_mask
    return base_mask_dict

def calc_next_half_size(cur_search_space_size, granu):
    if granu > cur_search_space_size:
        granu = 1
    tmp_small = granu
    tmp_large = granu * 2
    while tmp_large * 2 <= cur_search_space_size:
        tmp_small *= 2
        tmp_large *= 2
    ratio_small = cur_search_space_size / tmp_small
    ratio_large = cur_search_space_size / tmp_large
    if ratio_small - 2 < 2 - ratio_large:
        return tmp_small
    else:
        return tmp_large

def tfm_prune_filters_and_save_by_mask(model, layer_to_masks, save_file, fc_layer_idxes,
                                      subsequent_strategy,
                                    layer_idx_to_follow_offset={},
                                      fc_neurons_per_kernel=None,
                                       result_deps=None,
                                       data_format=None):
    assert data_format in ['NCHW', 'NHWC']

    result = dict()
    assert model.params.num_gpus == 1

    if subsequent_strategy is None:
        subsequent_map = None
    elif subsequent_strategy == 'simple':
        subsequent_map = {idx : (idx+1) for idx in layer_to_masks.keys()}
    else:
        subsequent_map = subsequent_strategy
    if type(fc_layer_idxes) is not list:
        fc_layer_idxes = [fc_layer_idxes]

    kernels = model.get_kernel_variables()

    for layer_idx, mask in layer_to_masks.items():

        kernel_tensor = kernels[layer_idx]
        print('cur kernel name:', kernel_tensor.name)
        bias_tensor = model.get_bias_variable_for_kernel(kernel_tensor)
        beta_tensor = model.get_beta_variable_for_kernel(kernel_tensor)
        gamma_tensor = model.get_gamma_variable_for_kernel(kernel_tensor)
        moving_mean_tensor = model.get_moving_mean_variable_for_kernel(kernel_tensor)
        moving_variance_tensor = model.get_moving_variance_variable_for_kernel(kernel_tensor)

        if kernel_tensor.name in result:
            kernel_value = result[kernel_tensor.name]
        else:
            kernel_value = model.get_value(kernel_tensor)

        #   no layer follows the current layer
        if subsequent_map is None or layer_idx not in subsequent_map:
            indexes_to_delete = np.where(mask == 0)[0]
        else:
            indexes_to_delete = np.where(mask == 0)[0]
            follows = subsequent_map[layer_idx]
            if type(follows) is not list:
                follows = [follows]
            for follow_idx in follows:
                follow_kernel_tensor = kernels[follow_idx]
                if follow_kernel_tensor.name in result:
                    kvf = result[follow_kernel_tensor.name]
                else:
                    kvf = model.get_value(follow_kernel_tensor)
                print('following kernel name: ', follow_kernel_tensor.name, 'origin shape: ', kvf.shape)
                if follow_idx in fc_layer_idxes:
                    offset = layer_idx_to_follow_offset.get(layer_idx, 0)
                    if offset > 0:
                        print('offset,',offset)
                    fc_indexes_to_delete = []
                    if fc_neurons_per_kernel is None:
                        conv_deps = kernel_value.shape[3] + offset
                        corresponding_neurons_per_kernel = kvf.shape[0] // conv_deps
                    else:
                        corresponding_neurons_per_kernel=fc_neurons_per_kernel
                        conv_deps = kvf.shape[0] // corresponding_neurons_per_kernel
                    print('total conv deps:', conv_deps, corresponding_neurons_per_kernel, 'neurons per kernel')
                    ###############     --------------------
                    if data_format == 'NHWC':
                        base = np.arange(offset, corresponding_neurons_per_kernel * conv_deps + offset, conv_deps)
                        for i in indexes_to_delete:
                            fc_indexes_to_delete.append(base + i)
                    else:
                        base = np.arange(offset, corresponding_neurons_per_kernel, 1)
                        for i in indexes_to_delete:
                            fc_indexes_to_delete.append(base + i * corresponding_neurons_per_kernel)
                    #################   --------------------
                    if len(fc_indexes_to_delete) > 0:
                        to_delete = np.concatenate(fc_indexes_to_delete, axis=0)
                        np.sort(to_delete)
                        print('prune these from the first fc: ', to_delete)
                        kvf = delete_or_keep(kvf, to_delete, axis=0)

                else:
                    offset = layer_idx_to_follow_offset.get(layer_idx, 0)
                    follow_indexes_to_delete = [offset + p for p in indexes_to_delete]
                    kvf = delete_or_keep(kvf, follow_indexes_to_delete, axis=2)
                result[follow_kernel_tensor.name] = kvf
                print('shape of pruned following kernel: ', kvf.shape)

        kernel_value_after_pruned = delete_or_keep(kernel_value, indexes_to_delete, axis=3)
        result[kernel_tensor.name] = kernel_value_after_pruned
        if result_deps is not None:
            result['deps'] = result_deps    #TODO
        if bias_tensor is not None:
            bias_value = delete_or_keep(model.get_value(bias_tensor), indexes_to_delete)
            result[bias_tensor.name] = bias_value
        if moving_mean_tensor is not None:
            moving_mean_value = delete_or_keep(model.get_value(moving_mean_tensor), indexes_to_delete)
            result[moving_mean_tensor.name] = moving_mean_value
        if moving_variance_tensor is not None:
            moving_variance_value = delete_or_keep(model.get_value(moving_variance_tensor), indexes_to_delete)
            result[moving_variance_tensor.name] = moving_variance_value
        if beta_tensor is not None:
            beta_value = delete_or_keep(model.get_value(beta_tensor), indexes_to_delete)
            result[beta_tensor.name] = beta_value
        if gamma_tensor is not None:
            gamma_value = delete_or_keep(model.get_value(gamma_tensor), indexes_to_delete)
            result[gamma_tensor.name] = gamma_value
        print('kernel name: ', kernel_tensor.name)
        print('removed filters by mask. shape of origin kernel {}, shape of pruned kernel {}'
            .format(kernel_value.shape, kernel_value_after_pruned.shape))

    key_variables = model.get_key_variables()
    for var in key_variables:
        if var.name not in result:
            result[var.name] = model.get_value(var)
    print('save {} varialbes to {} after pruning filters'.format(len(result), save_file))
    save_hdf5(result, save_file)