import numpy as np

from keras.engine import merge
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        UpSampling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Input
from keras.models import Model


def style_transfer(weights_path=None):
    input = Input(shape=(3, 256, 256), name='input', dtype='float32')

    # Downsampling
    c11 = Convolution2D(32, 9, 9, 
        init='uniform', subsample=(1, 1), border_mode='same', activation='linear')(input)
    bn11 = BatchNormalization(axis=1)(c11)
    a11 = Activation('relu')(bn11)

    c12 = Convolution2D(64, 3, 3, 
        init='uniform', subsample=(2, 2),  border_mode='same', activation='linear')(a11)
    bn12 = BatchNormalization(axis=1)(c12)
    a12 = Activation('relu')(bn12)

    c13 = Convolution2D(128, 3, 3, 
        init='uniform', subsample=(2, 2), border_mode='same', activation='linear')(a12)
    bn13 = BatchNormalization(axis=1)(c13)
    a13 = Activation('relu')(bn13)


    c21 = Convolution2D(128, 3, 3, 
        init='uniform', subsample=(1, 1), border_mode='same', activation='linear')(a13)
    bn21 = BatchNormalization(axis=1)(c21)
    a21 = Activation('relu')(bn21)
    c22 = Convolution2D(128, 3, 3, 
        init='uniform', subsample=(1, 1), border_mode='same', activation='linear')(a21)
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
    # out3 = Activation('relu')(bn32)
    out3 = merge([out2, bn32], mode='sum')

    c41 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out3)
    bn41 = BatchNormalization(axis=1)(c41)
    a41 = Activation('relu')(bn41)
    c42 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a41)
    bn42 = BatchNormalization(axis=1)(c42)
    # out4 = Activation('relu')(bn42)
    out4 = merge([out3, bn42], mode='sum')

    c51 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out4)
    bn51 = BatchNormalization(axis=1)(c51)
    a51 = Activation('relu')(bn51)
    c52 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a51)
    bn52 = BatchNormalization(axis=1)(c52)
    # out5 = Activation('relu')(bn52)
    out5 = merge([out4, bn52], mode='sum')

    c61 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out5)
    bn61 = BatchNormalization(axis=1)(c61)
    a61 = Activation('relu')(bn61)
    c62 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a61)
    bn62 = BatchNormalization(axis=1)(c62)
    # out6 = Activation('relu')(bn62)
    out6 = merge([out5, bn62], mode='sum')

    # This has to be checked, it might not be what we want
    c71 = Convolution2D(64, 1, 1, 
        init='uniform', subsample=(1, 1), border_mode='same', activation='linear')(out6)
    bn71 = BatchNormalization(axis=1)(c71)
    a71 = Activation('relu')(bn71)
    u71 = UpSampling2D(size=(2, 2))(a71)

    c81 = Convolution2D(32, 1, 1, 
        init='uniform', subsample=(1, 1), border_mode='same', activation='linear')(u71)
    bn81 = BatchNormalization(axis=1)(c81)
    a81 = Activation('relu')(bn81)
    u81 = UpSampling2D(size=(2, 2))(a81)    

    c91 = Convolution2D(3, 9, 9, 
        init='uniform', subsample=(1, 1), border_mode='same', activation='relu')(u81)
    #bn91 = BatchNormalization(axis=1)(c91)
    #a91 = Activation('relu')(bn91)
    
    model = Model(input=[input], output=[c91])

    if weights_path:
        model.load_weights(weights_path)

    return model
