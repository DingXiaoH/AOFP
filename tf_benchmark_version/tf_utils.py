import tensorflow as tf
import numpy as np
import re
import h5py

# VARIABLE_SAVE_BLACK_LIST = ['Adam', 'Exponential', 'global', 'step', 'power', 'Momentum']
# def is_variable_worth_saving(variable):
#     for keyword in VARIABLE_SAVE_BLACK_LIST:
#         if keyword in variable.name:
#             return False
#     return True
import time
import os

def latest_checkpoint_abs_path(train_dir):
    ckpt_file = os.path.join(train_dir, 'checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline().strip()
    ckpt_name = first_line.split(' ')[1].replace('"','')
    return os.path.join(train_dir, ckpt_name)

def tf_conv_flattened_corresponding_fc_idxs(nb_filters_last_conv, v_shape, filter_idxs):
    base = np.arange(0, v_shape[0], nb_filters_last_conv)
    result = []
    for f in filter_idxs:
        result.append(base + f)
    result = np.concatenate(result)
    return np.sort(result)


def check_and_get_deps_of_eqcls(eqcls_dict):
    origin_deps = np.zeros(len(eqcls_dict))
    target_deps = np.zeros(len(eqcls_dict))
    for layer_idx, eqcls in eqcls_dict.items():
        filter_idx_set = set()
        num_filters = 0
        for eqc in eqcls:
            num_filters += len(eqc)
            filter_idx_set.update(set(eqc))
        assert len(filter_idx_set) == num_filters
        origin_deps[layer_idx] = num_filters
        target_deps[layer_idx] = len(eqcls)
    return origin_deps, target_deps


def extract_deps_from_weights_file(file_path):
    if file_path.endswith('npy'):
        weight_dic = np.load(file_path).item()
    else:
        weight_dic = read_hdf5(file_path)
    if 'deps' in weight_dic:
        return weight_dic['deps']
    else:
        return None


def cur_time():
    return time.strftime('%Y,%b,%d,%X')

def convert_log_deps_to_list(log):
    ss = log.replace('\n', ' ').replace('[', '').replace(']', '').replace(',','').replace('  ', ' ').strip().split(' ')
    result = []
    for s in ss:
        if len(s.strip()) > 0:
            result.append(int(s))
    return result

def parse_int_str_to_array(s):
    s = s.replace('[ ', '').replace(']', '').replace('\n', ' ').replace('  ', ' ')
    ss = s.split(' ')
    result = []
    for d in ss:
        if d != '':
            result.append(int(d))
    return np.array(result, dtype=np.int32)

def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def save_hdf5(numpy_dict, file_path):
    with h5py.File(file_path, 'w') as f:
        for k,v in numpy_dict.items():
            f.create_dataset(str(k).replace('/','+'), data=v)
    print('saved {} arrays to {}'.format(len(numpy_dict), file_path))

def read_hdf5(file_path):
    result = {}
    with h5py.File(file_path, 'r') as f:
        for k in f.keys():
            value = np.asarray(f[k])
            if representsInt(k):
                result[int(k)] = value
            else:
                result[str(k).replace('+','/')] = value
    print('read {} arrays from {}'.format(len(result), file_path))
    return result

def append_to_log(file, log):
    with open(file, 'a') as f:
        print(log, file=f)

def extract_values_by_keyword_of_key_from_dict(dict, keyword):
    result = {}
    for k, v in dict.items():
        if keyword in k:
            result[k] = v
    return result

def extract_kernel_weights_from_np_dict(np_dict):
    return extract_values_by_keyword_of_key_from_dict(np_dict, 'kernel')

def extract_bias_weights_from_np_dict(np_dict):
    return extract_values_by_keyword_of_key_from_dict(np_dict, 'bias')


def tf_release():
    tf.reset_default_graph()

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def log_important(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        print(message, file=f)

def save_outs_and_labels(outs, labels, file_path):
    dic = {'outs': outs, 'labels':labels}
    np.save(file_path, dic)
    print('save outs and labels, path: ', file_path)

def load_outs_and_labels(file_path):
    if '{}' in file_path:
        outs = np.load(file_path.format('outs'))
        labels = np.load(file_path.format('labels'))
    else:
        dic = np.load(file_path).item()
        outs = dic['outs']
        labels = dic['labels']
    return outs, labels


# keras implementation : normal_loss + sum(square(every tensor))
# tf.contrib.layers implementation: normal_loss + 1/2 * sum(square(every tensor))
def calculate_l2_loss(tensors):
    l2_sum = 0.
    for t in tensors:
        l2_sum += 0.5 * tf.reduce_sum(tf.square(t))
    return l2_sum



def is_keywords_included(text, keywords):
    for k in keywords:
        if k in text:
            return True
    return False

def eliminate_all_patterns_and_starting_vs(text, patterns):
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    if text[:3] in ['v0/','v1/','v2/','v3/','v4/','v5/','v6/','v7/']:
        return text[3:]
    else:
        return text

def save_graph_weights_to_np(file_path, black_list=['Adam', 'Exponential', 'global', 'step', 'power', 'Momentum']):
    weights = tf.global_variables()
    cnt = 0
    dic = {}
    for t in weights:
        name = t.name
        if is_keywords_included(name, black_list):
            continue
        dic[name] = get_value(t)
        cnt += 1
    np.save(file_path, dic)
    print('save {} weights to {}'.format(cnt, file_path))

def double_keyword_gradients(origin_gradients, keywords):
    if keywords is None:
        return origin_gradients
    double_cnt = 0
    result = []
    print('---------------------- doubling gradients which contain keyword {}'.format(keywords))
    for grad, var in origin_gradients:
        need_double = False
        for kw in keywords:
            if kw in var.name:
                need_double = True
                break
        if need_double:
            result.append((2 * grad, var))
            double_cnt += 1
        else:
            result.append((grad, var))
    print('doubled gradients for {} bias variables'.format(double_cnt))
    return result

def double_bias_gradients(origin_gradients):
    bias_cnt = 0
    result = []
    print('doubling bias gradients')
    for grad, var in origin_gradients:
        if 'bias' in var.name:
            result.append((2 * grad, var))
            bias_cnt += 1
        else:
            result.append((grad, var))
    print('doubled gradients for {} bias variables'.format(bias_cnt))
    return result

def tf_extract_kernel_tensors():
    result = []
    for t in tf.global_variables():
        if 'kernel' in t.name:
            result.append(t)
    return result

def tf_extract_bias_tensors():
    result = []
    for t in tf.global_variables():
        if 'bias' in t.name:
            result.append(t)
    return result

def tf_get_gradients_by_idx(grads_and_vars, idx, keyword):
    kernel_seen = 0
    for (g, v) in grads_and_vars:
        if keyword in v.name:
            if kernel_seen == idx:
                return g
            else:
                kernel_seen += 1
    return None

def tf_get_gradient_by_var(grads_and_vars, var):
    for (g, v) in grads_and_vars:
        if v.name == var.name:
            return g
    return None


def get_value(t):
    return tf.get_default_session().run(t)

def set_value(t, value):
    ph = tf.placeholder(t.dtype, t.shape)
    op = t.assign(ph)
    tf.get_default_session().run(op, feed_dict={ph: value})


def assign_vars_from_np_dict_by_name(vars, init_np, ignore_patterns=['tower_[0-9]/'], replace_map={'fc1':'fc1-vc10', 'fc2':'fc2-vc10', 'error':'predictions'}, black_list=['Momentum','Adam', 'Exponential', 'global', 'step', 'power']):
    _dic = np.load(init_np).item()
    dic = {}
    for k, v in _dic.items():
        dic[eliminate_all_patterns(k, ignore_patterns)] = v
    cnt = 0
    for t in vars:
        name = eliminate_all_patterns(t.name, ignore_patterns)
        if is_keywords_included(name, black_list):
            print('not assigned because it is in blacklist:', name)
            continue
        if name in dic:
            set_value(t, dic[name])
            print(name)
            # print(np.sum(dic[name]))
            cnt += 1
        else:
            assigned = False
            for k, v in replace_map.items():
                replaced = name.replace(k, v)
                if replaced in dic:
                    set_value(t, dic[replaced])
                    assigned = True
                    cnt += 1
                    # print(replaced)
                    # print(np.sum(dic[replaced]))
            if not assigned:
                print('cannot find matched value for variable ', name)
    print('successfully loaded np. {} global variables, {} tensors assigned'.format(len(vars), cnt))