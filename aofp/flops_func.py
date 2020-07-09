import numpy as np

def get_con_flops(input_deps, output_deps, h, w=None, kernel_size=3, groups=1):
    if w is None:
        w = h
    return input_deps * output_deps * h * w * kernel_size * kernel_size // groups

def calculate_lenet5bn_flops(deps):
    assert len(deps) == 2
    result = []
    result.append(get_con_flops(1, deps[0], 24, 24, kernel_size=5))
    result.append(get_con_flops(deps[0], deps[1], 8, 8, kernel_size=5))
    # result.append(deps[1] * 16 * 500)
    # result.append(500 * 10)
    return np.sum(np.array(result, dtype=np.float32))


def calculate_vc_flops(deps):
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
    return np.sum(np.array(result, dtype=np.float32))


def calculate_rc_flops(deps, rc_n):
    result = []
    result.append(get_con_flops(3, deps[0], 32, 32))
    for i in range(rc_n):
        result.append(get_con_flops(deps[2*i], deps[2*i+1], 32, 32))
        result.append(get_con_flops(deps[2*i+1], deps[2*i+2], 32, 32))

    project_layer_idx = 2 * rc_n + 1
    result.append(get_con_flops(deps[project_layer_idx - 1], deps[project_layer_idx], 16, 16, 2))
    result.append(get_con_flops(deps[project_layer_idx - 1], deps[project_layer_idx + 1], 16, 16))
    result.append(get_con_flops(deps[project_layer_idx + 1], deps[project_layer_idx + 2], 16, 16))
    for i in range(rc_n - 1):
        result.append(get_con_flops(deps[2 * i + project_layer_idx + 2], deps[2 * i + project_layer_idx + 3], 16, 16))
        result.append(get_con_flops(deps[2 * i + project_layer_idx + 3], deps[2 * i + project_layer_idx + 4], 16, 16))

    project_layer_idx += 2 * rc_n + 1
    result.append(get_con_flops(deps[project_layer_idx - 1], deps[project_layer_idx], 8, 8, 2))
    result.append(get_con_flops(deps[project_layer_idx - 1], deps[project_layer_idx + 1], 8, 8))
    result.append(get_con_flops(deps[project_layer_idx + 1], deps[project_layer_idx + 2], 8, 8))
    for i in range(rc_n - 1):
        result.append(get_con_flops(deps[2 * i + project_layer_idx + 2], deps[2 * i + project_layer_idx + 3], 8, 8))
        result.append(get_con_flops(deps[2 * i + project_layer_idx + 3], deps[2 * i + project_layer_idx + 4], 8, 8))

    result.append(10*deps[-1])
    return np.sum(result)

def calculate_rc56_flops(deps):
    return calculate_rc_flops(deps, 9)