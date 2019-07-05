from models import model
import tensorflow as tf
slim = tf.contrib.slim

class VCModel(model.CNNModel):
    """Alexnet cnn model for cifar datasets.

    The model architecture follows the one defined in the tensorflow tutorial
    model.

    Reference model: tensorflow/models/tutorials/image/cifar10/cifar10.py
    Paper: http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    """

    def __init__(self, params=None):
        super(VCModel, self).__init__(
            'vc', 32, 64, 0.1, params=params)
        self.deps = params.deps
        
    def vc_conv(self, dep):
        self.cnn.conv(dep,
            mode='SAME',
            k_height=3,
            k_width=3,
            d_height=1,
            d_width=1,
            use_batch_norm=True,
            activation='relu',
            kernel_initializer=tf.contrib.layers.xavier_initializer())

    def add_inference(self, cnn):
        self.cnn = cnn
        # self.cnn.set_default_batch_norm_config(decay=0.999, epsilon=1e-3, scale=True)
        self.vc_conv(self.deps[0])
        self.vc_conv(self.deps[1])
        cnn.mpool(2, 2)
        self.vc_conv(self.deps[2])
        self.vc_conv(self.deps[3])
        cnn.mpool(2, 2)
        self.vc_conv(self.deps[4])
        self.vc_conv(self.deps[5])
        self.vc_conv(self.deps[6])
        cnn.mpool(2, 2)
        self.vc_conv(self.deps[7])
        self.vc_conv(self.deps[8])
        self.vc_conv(self.deps[9])
        cnn.mpool(2, 2)
        self.vc_conv(self.deps[10])
        self.vc_conv(self.deps[11])
        self.vc_conv(self.deps[12])
        cnn.mpool(2, 2)
        cnn.flatten()
        cnn.affine(512)

 