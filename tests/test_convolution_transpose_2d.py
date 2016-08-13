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

    def test_convolution_transpose_th(self):
        border_mode = 'valid'

        K.set_image_dim_ordering('th')
        batch = 1
        height = 2
        width = 2
        channels_in = 1
        channels_out = 2
        kernel_size = 3
        strides = (1, 1)
        input_shape = (channels_in, height, width)

        input = Input(shape=input_shape, dtype=K.floatx())
        conv_layer = ConvolutionTranspose2D(channels_out, kernel_size, kernel_size, 
                dim_ordering=K.image_dim_ordering(), init='one', 
                subsample=strides, border_mode=border_mode, activation='linear')
        output = conv_layer(input)
        model = Model(input=[input], output=[output])
        model.compile(loss='mean_squared_error', optimizer='sgd')

        x = np.ones((batch,) + input_shape).astype(K.floatx())
        kernel = conv_layer.W
        output_model = model.predict(x)
        if K._BACKEND == 'theano':
            output_shape = conv_layer.get_output_shape_for(K.shape(x))
            y = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(theano.shared(x), kernel, output_shape, 
                    filter_shape=None, border_mode=border_mode, subsample=strides, filter_flip=True)
            output = y.eval()
        else:
            sess = K.get_session()
            output_shape = conv_layer.get_output_shape_for(K.shape(x))
            output_shape = tf.pack([1, output_shape[2], output_shape[3], output_shape[1]])
            x = tf.transpose(x, (0, 2, 3, 1))
            kernel = tf.transpose(kernel, (2, 3, 1, 0))
            y = tf.nn.conv2d_transpose(x, kernel, output_shape, (1, ) + strides + (1, ), padding=border_mode.upper())
            y = tf.transpose(y, (0, 3, 1, 2))
            output = sess.run(y)
            
        self.assertEqual(output_model.shape, (1, 2, 4, 4))    
        self.assertEqual(output.shape, (1, 2, 4, 4))
        self.assertEqual(True, (output==output_model).all())

        # model.fit(x, x + 1, nb_epoch=1)

    def test_convolution_transpose_tf(self):
        border_mode = 'valid'

        K.set_image_dim_ordering('tf')
        batch = 1
        height = 2
        width = 2
        channels_in = 1
        channels_out = 2
        kernel_size = 3
        strides = (1, 1)
        input_shape = (height, width, channels_in)

        input = Input(shape=input_shape, dtype=K.floatx())
        conv_layer = ConvolutionTranspose2D(channels_out, kernel_size, kernel_size, 
                dim_ordering=K.image_dim_ordering(), init='one', 
                subsample=strides, border_mode=border_mode, activation='linear')
        output = conv_layer(input)
        model = Model(input=[input], output=[output])
        model.compile(loss='mean_squared_error', optimizer='sgd')

        x = np.ones((batch,) + input_shape).astype(K.floatx())
        kernel = conv_layer.W
        output_model = model.predict(x)
        if K._BACKEND == 'theano':
            output_shape = conv_layer.get_output_shape_for(K.shape(x))
            output_shape = (1, output_shape[3], output_shape[1], output_shape[2])
            x = np.transpose(x, (0, 3, 1, 2))
            kernel = T.transpose(kernel, (3, 2, 1, 0))
            y = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(theano.shared(x), kernel, output_shape, 
                    filter_shape=None, border_mode=border_mode, subsample=strides, filter_flip=True)
            y = T.transpose(y, (0, 2, 3, 1))
            output = y.eval()
        else:
            sess = K.get_session()
            output_shape = conv_layer.get_output_shape_for(K.shape(x))
            output_shape = tf.pack([1, output_shape[1], output_shape[2], output_shape[3]])
            y = tf.nn.conv2d_transpose(x, kernel, output_shape, (1, ) + strides + (1, ), padding=border_mode.upper())
            
            output = sess.run(y)
        
        self.assertEqual(output_model.shape, (1, 4, 4, 2))    
        self.assertEqual(output.shape, (1, 4, 4, 2))
        self.assertEqual(True, (output==output_model).all())

        # model.fit(x, x + 1, nb_epoch=1)
        
if __name__ == "__main__":
    unittest.main()