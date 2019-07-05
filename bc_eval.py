import sys
from bc_params_factory import default_params_for_eval, default_my_params
from bc import BenchmarkCNN
from bc_helpers import setup
from bc_constants import *
from tf_utils import extract_deps_from_weights_file
from bc_params_factory import decide_default_eval_batch_size, decide_default_eval_num_batches
from bc_util import AnonymousCNNModel

if __name__ == '__main__':
    model_name = sys.argv[1]
    init = sys.argv[2]

    # input_rotation = sys.argv[3]
    input_rotation = 0

    data_format = 'NHWC'
    force_subset = None
    num_batches = None

    # if len(sys.argv) >=4:
    #     data_format = sys.argv[3]
    # else:
    #     data_format = 'NHWC'
    # if len(sys.argv) >=5:
    #     force_subset = sys.argv[4]
    # else:
    #     force_subset = None
    # if force_subset == 'train':
    #     # num_batches = int(1281167 / decide_default_eval_batch_size(model_name))
    #     num_batches = 500
    #     print('eval on the train set')
    # else:
    #     num_batches = None



    if '035' in model_name:
        model_name = model_name.replace('035', '')
        multiplier = 0.35
    elif '050' in model_name:
        model_name = model_name.replace('050', '')
        multiplier = 0.5
    elif '075' in model_name:
        model_name = model_name.replace('075', '')
        multiplier = 0.75
    else:
        model_name = model_name
        multiplier = 1


    if model_name in ['vc','vh']:
        default_deps = VGG_ORIGIN_DEPS
    elif model_name == 'cfqkbn':
        default_deps = CFQK_ORIGIN_DEPS
    elif model_name == 'alexnet':
        default_deps = np.array([64, 192, 384, 384, 256])
    elif model_name == 'resnet50':
        default_deps = RESNET50_ORIGIN_DEPS_FLATTENED
    else:
        default_deps = None

    if '.hdf5' in init:
        deps = extract_deps_from_weights_file(init)
    else:
        deps = None

    if deps is None:
        deps = default_deps

    params = default_params_for_eval(model_name=model_name, data_format=data_format, depth_multiplier=multiplier, deps=deps)
    if '.hdf5' in init:
        my_params = default_my_params(init_hdf5=init, show_variables=True, load_ckpt=None, force_subset=force_subset,
            use_dense_layer=(model_name in ['vc', 'vh', 'alexClassic', 'vgg16', 'resnet18']))
    else:
        my_params = default_my_params(init_hdf5=None, show_variables=True, save_hdf5='test.hdf5', load_ckpt=init, force_subset=force_subset,
            use_dense_layer=False, input_rotation=input_rotation)

    if model_name == 'rh164_builder':
        from tfm_builder_rc import RH164Builder
        builder = RH164Builder(training=False, deps=deps)
        model = AnonymousCNNModel(model_name=model_name, image_size=32, batch_size=64, lr=0.05, tfm_builder=builder, params=params)
    elif model_name == 'rc164_builder':
        from tfm_builder_rc import RC164Builder
        builder = RC164Builder(training=False, deps=deps)
        params = params._replace(input_preprocessor='old')
        model = AnonymousCNNModel(model_name=model_name, image_size=32, batch_size=64, lr=0.05, tfm_builder=builder, params=params)
    else:
        model = None

    bc = BenchmarkCNN(params=params, my_params=my_params, model=model)
    bc.run()
