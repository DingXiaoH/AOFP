import torch.nn as nn
from custom_layers.flatten_layer import FlattenLayer

class AOFPMaxPool2d(nn.MaxPool2d):

    def __init__(self, kernel_size, stride, padding):
        super(AOFPMaxPool2d, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, inputs):
        if type(inputs) is dict:
            keys = list(inputs.keys())
            # print('keys to maxpool2d', keys)
            for k in keys:
                inputs[k] = super(AOFPMaxPool2d, self).forward(inputs[k])
            return inputs
        else:
            return super(AOFPMaxPool2d, self).forward(inputs)


class AOFPFlatten(FlattenLayer):

    def __init__(self):
        super(AOFPFlatten, self).__init__()

    def forward(self, inputs):
        if type(inputs) is dict:
            keys = list(inputs.keys())
            # print('keys to maxpool2d', keys)
            for k in keys:
                inputs[k] = super(AOFPFlatten, self).forward(inputs[k])
            return inputs
        else:
            return super(AOFPFlatten, self).forward(inputs)