from keras import backend as K
from keras.engine import merge
from keras.layers.convolutional import (Convolution2D, UpSampling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Input
from keras.models import Model

from models.layers.ConvolutionTranspose2D import ConvolutionTranspose2D

# inputs th ordering, BGR
def style_transfer_conv_transpose(weights_path=None, input_shape=(3, 600, 600)):
    input = Input(shape=input_shape, name='input', dtype='float32')

    # Downsampling
    c11 = Convolution2D(32, 9, 9, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    bn11 = BatchNormalization(axis=1)(c11)
    a11 = Activation('relu')(bn11)

    c12 = Convolution2D(64, 3, 3, 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(a11)
    bn12 = BatchNormalization(axis=1)(c12)
    a12 = Activation('relu')(bn12)

    c13 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a12)
    bn13 = BatchNormalization(axis=1)(c13)
    a13 = Activation('relu')(bn13)


    c21 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a13)
    bn21 = BatchNormalization(axis=1)(c21)
    a21 = Activation('relu')(bn21)
    c22 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a21)
    bn22 = BatchNormalization(axis=1)(c22)
    # a22 = Activation('relu')(bn22)
    out2 = merge([a13, bn22], mode='sum')

    c31 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out2)
    bn31 = BatchNormalization(axis=1)(c31)
    a31 = Activation('relu')(bn31)
    c32 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a31)
    bn32 = BatchNormalization(axis=1)(c32)
    # a32 = Activation('relu')(bn32)
    out3 = merge([out2, bn32], mode='sum')

    c41 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out3)
    bn41 = BatchNormalization(axis=1)(c41)
    a41 = Activation('relu')(bn41)
    c42 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a41)
    bn42 = BatchNormalization(axis=1)(c42)
    # a42 = Activation('relu')(bn42)
    out4 = merge([out3, bn42], mode='sum')

    c51 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out4)
    bn51 = BatchNormalization(axis=1)(c51)
    a51 = Activation('relu')(bn51)
    c52 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a51)
    bn52 = BatchNormalization(axis=1)(c52)
    # a52 = Activation('relu')(bn52)
    out5 = merge([out4, bn52], mode='sum')

    c61 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out5)
    bn61 = BatchNormalization(axis=1)(c61)
    a61 = Activation('relu')(bn61)
    c62 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a61)
    bn62 = BatchNormalization(axis=1)(c62)
    # a62 = Activation('relu')(bn62)
    out6 = merge([out5, bn62], mode='sum')

    ct71 = ConvolutionTranspose2D(64, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(out6)
    bn71 = BatchNormalization(axis=1)(ct71)
    a71 = Activation('relu')(bn71)
    
    ct81 = ConvolutionTranspose2D(32, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a71)
    bn81 = BatchNormalization(axis=1)(ct81)
    a81 = Activation('relu')(bn81)    

    c91 = ConvolutionTranspose2D(3, 9, 9, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a81)
    c92 = Activation(lambda x: 255 * K.tanh(x), name="output")(c91)    

    
    model = Model(input=[input], output=[c92])

    if weights_path:
        model.load_weights(weights_path)

    return model
    