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

border_mode = 'valid'

batch = 1
height = 2
width = 2
channels_in = 1
channels_out = 3
kernel_size = 3
strides = (1, 1)

input_shape = (channels_in, height, width)
kernel_shape = (channels_in, channels_out, kernel_size, kernel_size)

input = Input(shape=input_shape, dtype=K.floatx())
conv_layer = ConvolutionTranspose2D(channels_out, kernel_size, kernel_size, 
        init='one', subsample=strides, border_mode=border_mode, activation='linear')
output = conv_layer(input)
model = Model(input=[input], output=[output])
model.compile(loss='mean_squared_error', optimizer='sgd')

x = np.ones((1,) + input_shape).astype(K.floatx())
kernel = conv_layer.W

output_model = model.predict(x)
if K._BACKEND == 'theano':
    y = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(theano.shared(x), kernel, (None,) + input_shape, 
            filter_shape=None, border_mode=border_mode, subsample=strides, filter_flip=True)
    output = y.eval()
else:
    output_shape_tensor = conv_layer.get_output_shape_for(x)
    y = tf.nn.conv2d_transpose(x, kernel, output_shape_tensor, 
            (1, ) + strides + (1, ), padding=border_mode.upper())
    sess = tf.Session()
    output = sess.run(y)
    

print('input shape: ', input_shape)
print('output shape: ', output.shape)
print(output)
print(output_model)

model.fit(x, x + 1, nb_epoch=1)