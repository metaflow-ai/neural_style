import numpy as np

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
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    bn11 = BatchNormalization()(c11)
    a11 = Activation('relu')(bn11)

    c12 = Convolution2D(64, 3, 3, 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(a11)
    bn12 = BatchNormalization()(c12)
    a12 = Activation('relu')(bn12)

    c13 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a12)
    bn13 = BatchNormalization()(c13)
    a13 = Activation('relu')(bn13)


    c21 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a13)
    bn21 = BatchNormalization()(c21)
    a21 = Activation('relu')(bn21)
    c22 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a21)
    bn22 = BatchNormalization()(c22)
    a22 = Activation('relu')(bn22)

    # c31 = Convolution2D(128, 3, 3, 
    #     init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a22)
    # bn31 = BatchNormalization()(c31)
    # a31 = Activation('relu')(bn31)
    # c32 = Convolution2D(128, 3, 3, 
    #     init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a31)
    # bn32 = BatchNormalization()(c32)
    # a32 = Activation('relu')(bn32)

    # c41 = Convolution2D(128, 3, 3, 
    #     init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a32)
    # bn41 = BatchNormalization()(c41)
    # a41 = Activation('relu')(bn41)
    # c42 = Convolution2D(128, 3, 3, 
    #     init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a41)
    # bn42 = BatchNormalization()(c42)
    # a42 = Activation('relu')(bn42)

    # c51 = Convolution2D(128, 3, 3, 
    #     init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a42)
    # bn51 = BatchNormalization()(c51)
    # a51 = Activation('relu')(bn51)
    # c52 = Convolution2D(128, 3, 3, 
    #     init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a51)
    # bn52 = BatchNormalization()(c52)
    # a52 = Activation('relu')(bn52)

    # c61 = Convolution2D(128, 3, 3, 
    #     init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a52)
    # bn61 = BatchNormalization()(c61)
    # a61 = Activation('relu')(bn61)
    # c62 = Convolution2D(128, 3, 3, 
    #     init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a61)
    # bn62 = BatchNormalization()(c62)
    # a62 = Activation('relu')(bn62)


    # This has to be checked, it might not be what we want
    c71 = Convolution2D(64, 1, 1, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a22)
    # c71 = Convolution2D(64, 1, 1, 
    #     init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a62)
    bn71 = BatchNormalization()(c71)
    a71 = Activation('relu')(bn71)
    u71 = UpSampling2D(size=(2, 2))(a71)

    c81 = Convolution2D(32, 1, 1, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u71)
    bn81 = BatchNormalization()(c81)
    a81 = Activation('relu')(bn81)
    u81 = UpSampling2D(size=(2, 2))(a81)    

    c91 = Convolution2D(3, 9, 9, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u81)
    bn91 = BatchNormalization()(c91)
    a91 = Activation('relu')(bn91)
    
    model = Model(input=[input], output=[a91])

    if weights_path:
        model.load_weights(weights_path)

    return model