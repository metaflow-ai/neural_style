import numpy as np

from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input
from keras.models import Model


def VGG_16_MO(weights_path=None):
    input = Input(shape=(3, 224, 224), name='input')

    zp11 = ZeroPadding2D((1, 1))(input)
    c11 = Convolution2D(64, 3, 3, activation='relu')(zp11)
    zp12 = ZeroPadding2D((1, 1))(c11)
    c12 = Convolution2D(64, 3, 3, activation='relu', name="convlayers1_output")(zp12)
    mp1 = MaxPooling2D((2, 2), strides=(2, 2))(c12)

    zp21 = ZeroPadding2D((1, 1))(mp1)
    c21 = Convolution2D(128, 3, 3, activation='relu')(zp21)
    zp22 = ZeroPadding2D((1, 1))(c21)
    c22 = Convolution2D(128, 3, 3, activation='relu', name="convlayers2_output")(zp22)
    mp2 = MaxPooling2D((2, 2), strides=(2, 2))(c22)

    zp31 = ZeroPadding2D((1, 1))(mp2)
    c31 = Convolution2D(256, 3, 3, activation='relu')(zp31)
    zp32 = ZeroPadding2D((1, 1))(c31)
    c32 = Convolution2D(256, 3, 3, activation='relu')(zp32)
    zp33 = ZeroPadding2D((1, 1))(c32)
    c33 = Convolution2D(256, 3, 3, activation='relu', name="convlayers3_output")(zp33)
    mp3 = MaxPooling2D((2, 2), strides=(2, 2))(c33)

    zp41 = ZeroPadding2D((1, 1))(mp3)
    c41 = Convolution2D(512, 3, 3, activation='relu')(zp41)
    zp42 = ZeroPadding2D((1, 1))(c41)
    c42 = Convolution2D(512, 3, 3, activation='relu')(zp42)
    zp43 = ZeroPadding2D((1, 1))(c42)
    c43 = Convolution2D(512, 3, 3, activation='relu', name="convlayers4_output")(zp43)
    mp4 = MaxPooling2D((2, 2), strides=(2, 2))(c43)

    zp51 = ZeroPadding2D((1, 1))(mp4)
    c51 = Convolution2D(512, 3, 3, activation='relu')(zp51)
    zp52 = ZeroPadding2D((1, 1))(c51)
    c52 = Convolution2D(512, 3, 3, activation='relu')(zp52)
    zp53 = ZeroPadding2D((1, 1))(c52)
    c53 = Convolution2D(512, 3, 3, activation='relu', name="convlayers5_output")(zp53)
    mp5 = MaxPooling2D((2, 2), strides=(2, 2))(c53)

    f61 = Flatten()(mp5)
    d61 = Dense(4096, activation='relu')(f61)
    dp61 = Dropout(0.5)(d61)
    d62 = Dense(4096, activation='relu')(dp61)
    dp62 = Dropout(0.5)(d62)
    d63 = Dense(1000, activation='softmax', name="main_output")(dp62)

    model = Model(input=[input], output=[c12, c22, c33, c43, c53, d63])

    if weights_path:
        model.load_weights(weights_path)

    return model