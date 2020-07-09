import numpy as np
from utils.misc import save_hdf5

def delete_or_keep(array, idxes, axis=None):
    if len(idxes) > 0:
        return np.delete(array, idxes, axis=axis)
    else:
        return array

def aofp_prune(model, origin_deps, succ_strategy, save_path):
    final_deps = np.array(origin_deps)
    print('origin deps ', final_deps)

    mask_value_dict = {}
    for submodule in model.modules():
        if hasattr(submodule, 'base_mask'):
            mask_value_dict[submodule.conv_idx] = submodule.base_mask.cpu().numpy()

    kernel_name_list = []
    save_dict = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu().numpy()
        if v.ndim in [2, 4]:
            kernel_name_list.append(k)
        save_dict[k] = v

    for conv_id, kernel_name in enumerate(kernel_name_list):
        kernel_value = save_dict[kernel_name]
        if kernel_value.ndim == 2 or conv_id not in mask_value_dict:
            continue
        kernel_name = kernel_name_list[conv_id]
        mu_name = kernel_name.replace('conv.weight', 'bn.running_mean')
        sigma_name = kernel_name.replace('conv.weight', 'bn.running_var')
        gamma_name = kernel_name.replace('conv.weight', 'bn.weight')
        beta_name = kernel_name.replace('conv.weight', 'bn.bias')
        idx_to_delete = np.where(mask_value_dict[conv_id] == 0)[0]
        print('on layer {}, prune {}, cur {}, delete {}'.format(conv_id, idx_to_delete, final_deps, len(idx_to_delete)))
        final_deps[conv_id] -= len(idx_to_delete)
        print('pruning these: ', idx_to_delete, 'remaining: ', final_deps)
        save_dict[kernel_name] = delete_or_keep(kernel_value, idx_to_delete, axis=0)
        save_dict[mu_name] = delete_or_keep(save_dict[mu_name], idx_to_delete)
        save_dict[sigma_name] = delete_or_keep(save_dict[sigma_name], idx_to_delete)
        save_dict[gamma_name] = delete_or_keep(save_dict[gamma_name], idx_to_delete)
        save_dict[beta_name] = delete_or_keep(save_dict[beta_name], idx_to_delete)

        if len(idx_to_delete) > 0 and conv_id in succ_strategy:
            followers = succ_strategy[conv_id]
            if type(followers) is not list:
                followers = [followers]
            for fo in followers:
                fo_kernel_name = kernel_name_list[fo]
                fo_value = save_dict[fo_kernel_name]
                if fo_value.ndim == 4:
                    fo_value = np.delete(fo_value, idx_to_delete, axis=1)
                else:
                    fc_idx_to_delete = []
                    num_filters = kernel_value.shape[0]
                    fc_neurons_per_conv_kernel = fo_value.shape[1] // num_filters
                    # print('{} filters, {} neurons per kernel'.format(num_filters, fc_neurons_per_conv_kernel))
                    # base = np.arange(0, fc_neurons_per_conv_kernel * num_filters, num_filters)
                    # for i in idx_to_delete:
                    #     fc_idx_to_delete.append(base + i)
                    for i in idx_to_delete:
                        fc_idx_to_delete.append(np.arange(i * fc_neurons_per_conv_kernel,
                                                          (i+1) * fc_neurons_per_conv_kernel))
                    if len(fc_idx_to_delete) > 0:
                        fo_value = np.delete(fo_value, np.concatenate(fc_idx_to_delete, axis=0), axis=1)
                save_dict[fo_kernel_name] = fo_value

    save_dict['deps'] = final_deps
    #
    final_dict = {k.replace('base_path.', ''): v for k, v in save_dict.items() if
                  'mask' not in k and 'accumulated_' not in k and 't_value' not in k and 'half_start_iter' not in k and 'search_space' not in k}
    save_hdf5(final_dict, save_path)
    print('---------------saved {} numpy arrays to {}---------------'.format(len(save_dict), save_path))
    return final_deps