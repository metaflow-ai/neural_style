from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers import Input
from keras.models import Model

def VGG_16_headless_5(weights_path=None, input_shape=(3, 256, 256), trainable=False):
    input = Input(shape=input_shape, name='input', dtype='float32')

    zp11 = ZeroPadding2D((1, 1), trainable=trainable)(input)
    c11 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable, name="conv_1_1")(zp11)
    zp12 = ZeroPadding2D((1, 1))(c11)
    c12 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable, name="conv_1_2")(zp12)
    mp1 = MaxPooling2D((2, 2), strides=(2, 2))(c12)

    zp21 = ZeroPadding2D((1, 1))(mp1)
    c21 = Convolution2D(128, 3, 3, activation='relu', trainable=trainable, name="conv_2_1")(zp21)
    zp22 = ZeroPadding2D((1, 1))(c21)
    c22 = Convolution2D(128, 3, 3, activation='relu', trainable=trainable, name="conv_2_2")(zp22)
    mp2 = MaxPooling2D((2, 2), strides=(2, 2))(c22)

    zp31 = ZeroPadding2D((1, 1))(mp2)
    c31 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_1")(zp31)
    zp32 = ZeroPadding2D((1, 1))(c31)
    c32 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_2")(zp32)
    zp33 = ZeroPadding2D((1, 1))(c32)
    c33 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_3")(zp33)
    mp3 = MaxPooling2D((2, 2), strides=(2, 2))(c33)

    zp41 = ZeroPadding2D((1, 1))(mp3)
    c41 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_4_1")(zp41)
    zp42 = ZeroPadding2D((1, 1))(c41)
    c42 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_4_2")(zp42)
    zp43 = ZeroPadding2D((1, 1))(c42)
    c43 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_4_3")(zp43)
    mp4 = MaxPooling2D((2, 2), strides=(2, 2))(c43)

    zp51 = ZeroPadding2D((1, 1))(mp4)
    c51 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_5_1")(zp51)
    zp52 = ZeroPadding2D((1, 1))(c51)
    c52 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_5_2")(zp52)
    zp53 = ZeroPadding2D((1, 1))(c52)
    c53 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_5_3")(zp53)

    model = Model(input=[input], output=[
        c11, c12, 
        c21, c22, 
        c31, c32, c33, 
        c41, c42, c43,
        c51, c52, c53,
        c33]
    )

    if weights_path:
        model.load_weights(weights_path)

    return model

def VGG_16_headless_4(weights_path=None, input_shape=(3, 256, 256), trainable=False):
    input = Input(shape=input_shape, name='input', dtype='float32')

    zp11 = ZeroPadding2D((1, 1), trainable=trainable)(input)
    c11 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable, name="conv_1_1")(zp11)
    zp12 = ZeroPadding2D((1, 1))(c11)
    c12 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable, name="conv_1_2")(zp12)
    mp1 = MaxPooling2D((2, 2), strides=(2, 2))(c12)

    zp21 = ZeroPadding2D((1, 1))(mp1)
    c21 = Convolution2D(128, 3, 3, activation='relu', trainable=trainable, name="conv_2_1")(zp21)
    zp22 = ZeroPadding2D((1, 1))(c21)
    c22 = Convolution2D(128, 3, 3, activation='relu', trainable=trainable, name="conv_2_2")(zp22)
    mp2 = MaxPooling2D((2, 2), strides=(2, 2))(c22)

    zp31 = ZeroPadding2D((1, 1))(mp2)
    c31 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_1")(zp31)
    zp32 = ZeroPadding2D((1, 1))(c31)
    c32 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_2")(zp32)
    zp33 = ZeroPadding2D((1, 1))(c32)
    c33 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_3")(zp33)
    mp3 = MaxPooling2D((2, 2), strides=(2, 2))(c33)

    zp41 = ZeroPadding2D((1, 1))(mp3)
    c41 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_4_1")(zp41)
    zp42 = ZeroPadding2D((1, 1))(c41)
    c42 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_4_2")(zp42)
    zp43 = ZeroPadding2D((1, 1))(c42)
    c43 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_4_3")(zp43)

    model = Model(input=[input], output=[
        c11, c12, 
        c21, c22, 
        c31, c32, c33, 
        c41, c42, c43]
    )

    if weights_path:
        model.load_weights(weights_path)

    return model

def VGG_16_headless_3(weights_path=None, input_shape=(3, 256, 256), trainable=False):
    input = Input(shape=input_shape, name='input', dtype='float32')

    zp11 = ZeroPadding2D((1, 1), trainable=trainable)(input)
    c11 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable, name="conv_1_1")(zp11)
    zp12 = ZeroPadding2D((1, 1))(c11)
    c12 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable, name="conv_1_2")(zp12)
    mp1 = MaxPooling2D((2, 2), strides=(2, 2))(c12)

    zp21 = ZeroPadding2D((1, 1))(mp1)
    c21 = Convolution2D(128, 3, 3, activation='relu', trainable=trainable, name="conv_2_1")(zp21)
    zp22 = ZeroPadding2D((1, 1))(c21)
    c22 = Convolution2D(128, 3, 3, activation='relu', trainable=trainable, name="conv_2_2")(zp22)
    mp2 = MaxPooling2D((2, 2), strides=(2, 2))(c22)

    zp31 = ZeroPadding2D((1, 1))(mp2)
    c31 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_1")(zp31)
    zp32 = ZeroPadding2D((1, 1))(c31)
    c32 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_2")(zp32)
    zp33 = ZeroPadding2D((1, 1))(c32)
    c33 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_3")(zp33)

    model = Model(input=[input], output=[
        c11, c12, 
        c21, c22, 
        c31, c32, c33]
    )

    if weights_path:
        model.load_weights(weights_path)

    return model

def VGG_16_headless_2(weights_path=None, input_shape=(3, 256, 256), trainable=False):
    input = Input(shape=input_shape, name='input', dtype='float32')

    zp11 = ZeroPadding2D((1, 1), trainable=trainable)(input)
    c11 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable, name="conv_1_1")(zp11)
    zp12 = ZeroPadding2D((1, 1))(c11)
    c12 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable, name="conv_1_2")(zp12)
    mp1 = MaxPooling2D((2, 2), strides=(2, 2))(c12)

    zp21 = ZeroPadding2D((1, 1))(mp1)
    c21 = Convolution2D(128, 3, 3, activation='relu', trainable=trainable, name="conv_2_1")(zp21)
    zp22 = ZeroPadding2D((1, 1))(c21)
    c22 = Convolution2D(128, 3, 3, activation='relu', trainable=trainable, name="conv_2_2")(zp22)

    model = Model(input=[input], output=[
        c11, c12, 
        c21, c22]
    )

    if weights_path:
        model.load_weights(weights_path)

    return model

def VGG_16_headless_1(weights_path=None, input_shape=(3, 256, 256), trainable=False):
    input = Input(shape=input_shape, name='input', dtype='float32')

    zp11 = ZeroPadding2D((1, 1), trainable=trainable)(input)
    c11 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable, name="conv_1_1")(zp11)
    zp12 = ZeroPadding2D((1, 1))(c11)
    c12 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable, name="conv_1_2")(zp12)

    model = Model(input=[input], output=[
        c11, c12]
    )

    if weights_path:
        model.load_weights(weights_path)

    return model