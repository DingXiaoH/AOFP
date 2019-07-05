from models.model import CNNModel
from tf_utils import read_hdf5, save_hdf5
import numpy as np
import re


def try_to_get_variable_by_eliminating_batchnorm_words(name_to_variables, desired_name):
    batchnorm_eliminated_name_to_variables = {}
    pattern = re.compile('batchnorm(\d+)/')
    for n, v in name_to_variables.items():
        batchnorm_eliminated_name_to_variables[re.sub(pattern, '', n)] = v
    return batchnorm_eliminated_name_to_variables.get(desired_name, None)


class AnonymousCNNModel(CNNModel):

    def __init__(self, model_name, image_size, batch_size, lr, tfm_builder, params):
        super(AnonymousCNNModel, self).__init__(model=model_name, image_size=image_size, batch_size=batch_size, learning_rate=lr, params=params)
        self.tfm_builder = tfm_builder

    def skip_final_affine_layer(self):
        return True

    def add_inference(self, cnn):
        cnn.top_layer = self.tfm_builder.build(cnn.top_layer)
        cnn.top_size = cnn.top_layer.shape[-1].value
        print('build by an AnonymousCNNModel and the tfm_builder')


#
# class SlimCNNModel(CNNModel):
#
#     def __init__(self, model_name, image_size, batch_size, lr, tfm_builder, params):
#         super(AnonymousCNNModel, self).__init__(model=model_name, image_size=image_size, batch_size=batch_size, learning_rate=lr, params=params)
#         self.tfm_builder = tfm_builder
#
#     def skip_final_affine_layer(self):
#         return True
#
#     def add_inference(self, cnn):
#         cnn.top_layer = self.tfm_builder.build(cnn.top_layer)
#         cnn.top_size = cnn.top_layer.shape[-1].value
#         print('build by an AnonymousCNNModel and the tfm_builder')




def anonymous_model(model_name, builder, image_size, batch_size, lr, params):
    return AnonymousCNNModel(model_name=model_name, tfm_builder=builder, image_size=image_size, batch_size=batch_size, lr=lr, params=params)


def convert_nchw_to_nwhc(old_hdf5, new_hdf5):
    old_weights = read_hdf5(old_hdf5)
    new_weights = {}
    for n, v in old_weights.items():
        if v.ndim == 4:
            print('transposing ', n)
            new_weights[n] = np.transpose(v, [0, 2, 3, 1])
        else:
            new_weights[n] = v
    save_hdf5(new_weights, new_hdf5)

