import os, sys, unittest
import numpy as np

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

from keras import backend as K
if K._BACKEND == 'theano':
    import theano
    import theano.tensor as T
else:
    import tensorflow as tf

from keras.layers import Input
from keras.models import Model

from models.layers.ConvolutionTranspose2D import ConvolutionTranspose2D

class TestImUtils(unittest.TestCase):

    def test_convolution_transpose(self):
        border_mode = 'valid'

        batch = 1
        height = 2
        width = 2
        channels_in = 1
        channels_out = 3
        kernel_size = 3
        strides = (1, 1)

        input_shape = (channels_in, height, width)

        input = Input(shape=input_shape, dtype=K.floatx())
        conv_layer = ConvolutionTranspose2D(channels_out, kernel_size, kernel_size, 
                init='one', subsample=strides, border_mode=border_mode, activation='linear')
        output = conv_layer(input)
        model = Model(input=[input], output=[output])
        model.compile(loss='mean_squared_error', optimizer='sgd')

        x = np.ones((batch,) + input_shape).astype(K.floatx())
        kernel = conv_layer.W

        output_model = model.predict(x)
        if K._BACKEND == 'theano':
            y = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(theano.shared(x), kernel, (None,) + input_shape, 
                    filter_shape=None, border_mode=border_mode, subsample=strides, filter_flip=True)
            output = y.eval()
        else:
            output_shape = conv_layer.get_output_shape_for(K.shape(x))
            output_shape_tensor = tf.pack([1, output_shape[2], output_shape[3], output_shape[1]])
            x = tf.transpose(x, (0, 2, 3, 1))
            kernel = tf.transpose(kernel, (2, 3, 1, 0))
            y = tf.nn.conv2d_transpose(x, kernel, output_shape_tensor, (1, ) + strides + (1, ), padding=border_mode.upper())
            y = tf.transpose(y, (0, 3, 1, 2))
            sess = K.get_session()
            output = sess.run(y)
            
        self.assertEqual(output.shape, (1, 3, 4, 4))
        self.assertEqual(True, (output==output_model).all())

        # model.fit(x, x + 1, nb_epoch=1)
        