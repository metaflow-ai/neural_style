from keras.engine import merge
from keras.layers.convolutional import (Convolution2D, UpSampling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Input
from keras.models import Model


def style_transfer(weights_path=None, input_shape=(3, 256, 256)):
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

    # This is not a deconvolution (but might be close enough)
    u71 = UpSampling2D(size=(2, 2))(out6)
    c71 = Convolution2D(64, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u71)
    bn71 = BatchNormalization(axis=1)(c71)
    a71 = Activation('relu')(bn71)
    
    u81 = UpSampling2D(size=(2, 2))(a71)
    c81 = Convolution2D(32, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u81)
    bn81 = BatchNormalization(axis=1)(c81)
    a81 = Activation('relu')(bn81)    

    c91 = Convolution2D(3, 9, 9, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='relu', name='output')(a81)
    
    model = Model(input=[input], output=[c91])

    if weights_path:
        model.load_weights(weights_path)

    return model


def style_transfer_one_middle(weights_path=None, input_shape=(3, 256, 256)):
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


    # This is not a deconvolution (but might be close enough)
    u71 = UpSampling2D(size=(2, 2))(out2)
    c71 = Convolution2D(64, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u71)
    bn71 = BatchNormalization(axis=1)(c71)
    a71 = Activation('relu')(bn71)
    
    u81 = UpSampling2D(size=(2, 2))(a71)
    c81 = Convolution2D(32, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u81)
    bn81 = BatchNormalization(axis=1)(c81)
    a81 = Activation('relu')(bn81)    

    c91 = Convolution2D(3, 9, 9, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='relu', name='output')(a81)
    
    model = Model(input=[input], output=[c91])

    if weights_path:
        model.load_weights(weights_path)

    return model


def style_transfer_3_3_only(weights_path=None, input_shape=(3, 256, 256)):
    input = Input(shape=input_shape, name='input', dtype='float32')

    # Downsampling
    c11 = Convolution2D(32, 3, 3, 
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

    # This is not a deconvolution (but might be close enough)
    u71 = UpSampling2D(size=(2, 2))(out6)
    c71 = Convolution2D(64, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u71)
    bn71 = BatchNormalization(axis=1)(c71)
    a71 = Activation('relu')(bn71)
    
    u81 = UpSampling2D(size=(2, 2))(a71)
    c81 = Convolution2D(32, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u81)
    bn81 = BatchNormalization(axis=1)(c81)
    a81 = Activation('relu')(bn81)    

    c91 = Convolution2D(3, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='relu', name='output')(a81)
    
    model = Model(input=[input], output=[c91])

    if weights_path:
        model.load_weights(weights_path)

    return model


def style_transfer_3_3_only_small_double_stride(weights_path=None, input_shape=(3, 256, 256)):
    input = Input(shape=input_shape, name='input', dtype='float32')

    # Downsampling
    c11 = Convolution2D(32, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(input)
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


    # This is not a deconvolution (but might be close enough)
    u71 = UpSampling2D(size=(2, 2))(out2)
    c71 = Convolution2D(64, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u71)
    bn71 = BatchNormalization(axis=1)(c71)
    a71 = Activation('relu')(bn71)
    
    u72 = UpSampling2D(size=(2, 2))(a71)
    c72 = Convolution2D(32, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u72)
    bn72 = BatchNormalization(axis=1)(c72)
    a72 = Activation('relu')(bn72)    

    u73 = UpSampling2D(size=(2, 2))(a72)
    c73 = Convolution2D(3, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='relu', name='output')(u73)
    
    model = Model(input=[input], output=[c73])

    if weights_path:
        model.load_weights(weights_path)

    return model

def style_transfer_3_3_only_double_stride(weights_path=None, input_shape=(3, 256, 256)):
    input = Input(shape=input_shape, name='input', dtype='float32')

    # Downsampling
    c11 = Convolution2D(32, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(input)
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

    # This is not a deconvolution (but might be close enough)
    u71 = UpSampling2D(size=(2, 2))(out6)
    c71 = Convolution2D(64, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u71)
    bn71 = BatchNormalization(axis=1)(c71)
    a71 = Activation('relu')(bn71)
    
    u81 = UpSampling2D(size=(2, 2))(a71)
    c81 = Convolution2D(32, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u81)
    bn81 = BatchNormalization(axis=1)(c81)
    a81 = Activation('relu')(bn81)    

    u91 = UpSampling2D(size=(2, 2))(a81)
    c92 = Convolution2D(3, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='relu', name='output')(u91)
    
    model = Model(input=[input], output=[c92])

    if weights_path:
        model.load_weights(weights_path)

    return model

def style_transfer_3_3_only_double_stride_nobatchnorm(weights_path=None, input_shape=(3, 256, 256)):
    input = Input(shape=input_shape, name='input', dtype='float32')

    # Downsampling
    c11 = Convolution2D(32, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(input)
    a11 = Activation('relu')(c11)

    c12 = Convolution2D(64, 3, 3, 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(a11)
    a12 = Activation('relu')(c12)

    c13 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a12)
    a13 = Activation('relu')(c13)


    c21 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a13)
    a21 = Activation('relu')(c21)
    c22 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a21)
    # a22 = Activation('relu')(c22)
    out2 = merge([a13, c22], mode='sum')

    c31 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out2)
    a31 = Activation('relu')(c31)
    c32 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a31)
    # out3 = Activation('relu')(c32)
    out3 = merge([out2, c32], mode='sum')

    c41 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out3)
    a41 = Activation('relu')(c41)
    c42 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a41)
    # out4 = Activation('relu')(c42)
    out4 = merge([out3, c42], mode='sum')

    c51 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out4)
    a51 = Activation('relu')(c51)
    c52 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a51)
    # out5 = Activation('relu')(c52)
    out5 = merge([out4, c52], mode='sum')

    c61 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out5)
    a61 = Activation('relu')(c61)
    c62 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a61)
    # out6 = Activation('relu')(c62)
    out6 = merge([out5, c62], mode='sum')

    # This is not a deconvolution (but might be close enough)
    u71 = UpSampling2D(size=(2, 2))(out6)
    c71 = Convolution2D(64, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u71)
    a71 = Activation('relu')(c71)
    
    u81 = UpSampling2D(size=(2, 2))(a71)
    c81 = Convolution2D(32, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(u81)
    a81 = Activation('relu')(c81)

    u91 = UpSampling2D(size=(2, 2))(a81)
    c92 = Convolution2D(3, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='relu', name='output')(u91)
    
    model = Model(input=[input], output=[c92])

    if weights_path:
        model.load_weights(weights_path)

    return model