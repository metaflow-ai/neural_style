import numpy as np
from keras import backend as K
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Lambda
from keras.layers import Input
from keras.models import Model

def grams(m):
    m_shaped = K.reshape(m, (m.shape[0], m.shape[1], m.shape[2] * m.shape[3]))

    return m_shaped

def grams_output_shape(input_shape):
    return (input_shape[0], input_shape[1], input_shape[2] * input_shape[3])

def VGG_16_headless_5(weights_path=None, trainable=True):
    input = Input(shape=(3, 256, 256), name='input', dtype='float32')

    zp11 = ZeroPadding2D((1, 1), trainable=trainable)(input)
    c11 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable)(zp11)
    zp12 = ZeroPadding2D((1, 1))(c11)
    c12 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable)(zp12)
    mp1 = MaxPooling2D((2, 2), strides=(2, 2))(c12)

    zp21 = ZeroPadding2D((1, 1))(mp1)
    c21 = Convolution2D(128, 3, 3, activation='relu', trainable=trainable)(zp21)
    zp22 = ZeroPadding2D((1, 1))(c21)
    c22 = Convolution2D(128, 3, 3, activation='relu', trainable=trainable)(zp22)
    mp2 = MaxPooling2D((2, 2), strides=(2, 2))(c22)

    zp31 = ZeroPadding2D((1, 1))(mp2)
    c31 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable)(zp31)
    zp32 = ZeroPadding2D((1, 1))(c31)
    c32 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable)(zp32)
    zp33 = ZeroPadding2D((1, 1))(c32)
    c33 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="out_feat3")(zp33)
    mp3 = MaxPooling2D((2, 2), strides=(2, 2))(c33)

    zp41 = ZeroPadding2D((1, 1))(mp3)
    c41 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable)(zp41)
    zp42 = ZeroPadding2D((1, 1))(c41)
    c42 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable)(zp42)
    zp43 = ZeroPadding2D((1, 1))(c42)
    c43 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable)(zp43)
    mp4 = MaxPooling2D((2, 2), strides=(2, 2))(c43)

    zp51 = ZeroPadding2D((1, 1))(mp4)
    c51 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable)(zp51)
    zp52 = ZeroPadding2D((1, 1))(c51)
    c52 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable)(zp52)
    zp53 = ZeroPadding2D((1, 1))(c52)
    c53 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable)(zp53)

    # Outputs
    out_style1 = Lambda(grams, output_shape=grams_output_shape, name="out_style1")(c12)
    out_style2 = Lambda(grams, output_shape=grams_output_shape, name="out_style2")(c22)
    out_style3 = Lambda(grams, output_shape=grams_output_shape, name="out_style3")(c33)
    out_style4 = Lambda(grams, output_shape=grams_output_shape, name="out_style4")(c43)
    out_style5 = Lambda(grams, output_shape=grams_output_shape, name="out_style5")(c53)

    model = Model(input=[input], output=[out_style1, out_style2, out_style3, c33, out_style4, out_style5])

    if weights_path:
        model.load_weights(weights_path)

    return model