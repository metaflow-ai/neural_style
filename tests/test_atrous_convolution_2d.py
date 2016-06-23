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

from models.layers.ATrousConvolution2D import ATrousConvolution2D

class TestATrousConvolution2d(unittest.TestCase):

    def test_convolution_transpose_th(self):
        if K._BACKEND != 'tensorflow':
            return True
        K.set_image_dim_ordering('th')
        
        border_mode = 'valid'
        batch = 1
        height = 10
        width = 10
        channels_in = 1
        channels_out = 2
        kernel_size = 3
        rate = 2
        input_shape = (channels_in, height, width)

        input = Input(shape=input_shape, dtype=K.floatx())
        conv_layer = ATrousConvolution2D(channels_out, kernel_size, kernel_size, 
                rate, dim_ordering=K.image_dim_ordering(), init='one', 
                border_mode=border_mode, activation='linear')
        output = conv_layer(input)
        model = Model(input=[input], output=[output])
        model.compile(loss='mean_squared_error', optimizer='sgd')

        x = np.ones((batch,) + input_shape).astype(K.floatx())
        kernel = conv_layer.W
        output_model = model.predict(x)
        if K._BACKEND == 'tensorflow':
            x = tf.transpose(x, (0, 2, 3, 1))
            kernel = tf.transpose(kernel, (2, 3, 1, 0))

            y = tf.nn.atrous_conv2d(x, kernel, rate, padding=border_mode.upper())

            y = tf.transpose(y, (0, 3, 1, 2))
            output = y.eval(session=K.get_session())
            
        self.assertEqual(output_model.shape, (1, 2, 6, 6))    
        self.assertEqual(output.shape, (1, 2, 6, 6))
        self.assertEqual(True, (output==output_model).all())

    def test_convolution_transpose_tf(self):
        if K._BACKEND != 'tensorflow':
            return True
        K.set_image_dim_ordering('tf')

        border_mode = 'valid'
        batch = 1
        height = 10
        width = 10
        channels_in = 1
        channels_out = 2
        kernel_size = 3
        # effective kernel size: kernel_size + (kernel_size - 1) * (rate - 1)
        rate = 2
        input_shape = (height, width, channels_in)

        input = Input(shape=input_shape, dtype=K.floatx())
        conv_layer = ATrousConvolution2D(channels_out, kernel_size, kernel_size, 
                rate, dim_ordering=K.image_dim_ordering(), init='one', 
                border_mode=border_mode, activation='linear')
        output = conv_layer(input)
        model = Model(input=[input], output=[output])
        model.compile(loss='mean_squared_error', optimizer='sgd')

        x = np.ones((batch,) + input_shape).astype(K.floatx())
        kernel = conv_layer.W
        output_model = model.predict(x)
        if K._BACKEND == 'tensorflow':
            y = tf.nn.atrous_conv2d(x, kernel, rate, padding=border_mode.upper())
            output = y.eval(session=K.get_session())
        
        self.assertEqual(output_model.shape, (1, 6, 6, 2))    
        self.assertEqual(output.shape, (1, 6, 6, 2))
        self.assertEqual(True, (output==output_model).all())

    def test_convolution_transpose_tf_sameborder(self):
        if K._BACKEND != 'tensorflow':
            return True
        K.set_image_dim_ordering('tf')

        border_mode = 'same'
        batch = 1
        height = 10
        width = 10
        channels_in = 1
        channels_out = 2
        kernel_size = 3
        # effective kernel size: kernel_size + (kernel_size - 1) * (rate - 1)
        rate = 2
        input_shape = (height, width, channels_in)

        input = Input(shape=input_shape, dtype=K.floatx())
        conv_layer = ATrousConvolution2D(channels_out, kernel_size, kernel_size, 
                rate, dim_ordering=K.image_dim_ordering(), init='one', 
                border_mode=border_mode, activation='linear')
        output = conv_layer(input)
        model = Model(input=[input], output=[output])
        model.compile(loss='mean_squared_error', optimizer='sgd')

        x = np.ones((batch,) + input_shape).astype(K.floatx())
        kernel = conv_layer.W
        output_model = model.predict(x)
        if K._BACKEND == 'tensorflow':
            y = tf.nn.atrous_conv2d(x, kernel, rate, padding=border_mode.upper())
            output = y.eval(session=K.get_session())
        
        self.assertEqual(output_model.shape, (1, 10, 10, 2))    
        self.assertEqual(output.shape, (1, 10, 10, 2))
        self.assertEqual(True, (output==output_model).all())

        
if __name__ == "__main__":
    unittest.main()