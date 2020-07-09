from tfm_constants import *

def get_con_flops(input_deps, output_deps, h, w=None, kernel_size=3):
    if w is None:
        w = h
    return input_deps * output_deps * h * w * kernel_size * kernel_size

def get_con_parameters(input_deps, output_deps, h, w=None, kernel_size=3):
    return input_deps * output_deps * kernel_size * kernel_size


def calculate_vfs_flops(deps):
    assert len(deps) == 13
    result = []
    result.append(get_con_flops(3, deps[0], 32, 32))
    result.append(get_con_flops(deps[0], deps[1], 32, 32))
    result.append(get_con_flops(deps[1], deps[2], 16, 16))
    result.append(get_con_flops(deps[2], deps[3], 16, 16))
    result.append(get_con_flops(deps[3], deps[4], 8, 8))
    result.append(get_con_flops(deps[4], deps[5], 8, 8))
    result.append(get_con_flops(deps[5], deps[6], 8, 8))
    result.append(get_con_flops(deps[6], deps[7], 4, 4))
    result.append(get_con_flops(deps[7], deps[8], 4, 4))
    result.append(get_con_flops(deps[8], deps[9], 4, 4))
    result.append(get_con_flops(deps[9], deps[10], 2, 2))
    result.append(get_con_flops(deps[10], deps[11], 2, 2))
    result.append(get_con_flops(deps[11], deps[12], 2, 2))
    result.append(deps[12] * 512)
    result.append(512 * 10)
    return np.array(result, dtype=np.float32)

def calculate_vfs_params(deps):
    assert len(deps) == 13
    result = []
    result.append(get_con_parameters(3, deps[0], 32, 32))
    result.append(get_con_parameters(deps[0], deps[1], 32, 32))
    result.append(get_con_parameters(deps[1], deps[2], 16, 16))
    result.append(get_con_parameters(deps[2], deps[3], 16, 16))
    result.append(get_con_parameters(deps[3], deps[4], 8, 8))
    result.append(get_con_parameters(deps[4], deps[5], 8, 8))
    result.append(get_con_parameters(deps[5], deps[6], 8, 8))
    result.append(get_con_parameters(deps[6], deps[7], 4, 4))
    result.append(get_con_parameters(deps[7], deps[8], 4, 4))
    result.append(get_con_parameters(deps[8], deps[9], 4, 4))
    result.append(get_con_parameters(deps[9], deps[10], 2, 2))
    result.append(get_con_parameters(deps[10], deps[11], 2, 2))
    result.append(get_con_parameters(deps[11], deps[12], 2, 2))
    result.append(deps[12] * 512)
    result.append(512 * 10)
    return np.array(result, dtype=np.float32)
