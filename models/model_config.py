# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model configurations for CNN benchmarks.
"""

from models import vc_model


_model_name_to_cifar_model = {
    'vc': vc_model.VCModel,
}

def _get_model_map(dataset_name):
    """Get name to model map for specified dataset."""
    if 'cifar10' in dataset_name:
        return _model_name_to_cifar_model
    else:
        raise ValueError('Invalid dataset name: %s' % dataset_name)


def get_model_config(model_name, dataset, params):
    """Map model name to model network configuration."""
    model_map = _get_model_map(dataset.name)
    if model_name not in model_map:
        raise ValueError('Invalid model name \'%s\' for dataset \'%s\'' %
                         (model_name, dataset.name))
    else:
        return model_map[model_name](params=params)


def register_model(model_name, dataset_name, model_func):
    """Register a new model that can be obtained with `get_model_config`."""
    model_map = _get_model_map(dataset_name)
    if model_name in model_map:
        raise ValueError('Model "%s" is already registered for dataset "%s"' %
                         (model_name, dataset_name))
    model_map[model_name] = model_func
