import re 

from keras.layers.convolutional import (Convolution2D, MaxPooling2D, AveragePooling2D,
                                        ZeroPadding2D)
from keras.layers import Input
from keras.models import Model

def VGG_19_headless_5(weights_path=None, input_shape=(3, 256, 256), trainable=False, poolingType='max'):
    input = Input(shape=input_shape, name='input', dtype='float32')

    zp11 = ZeroPadding2D((1, 1), trainable=trainable)(input)
    c11 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable, name="conv_1_1")(zp11)
    zp12 = ZeroPadding2D((1, 1))(c11)
    c12 = Convolution2D(64, 3, 3, activation='relu', trainable=trainable, name="conv_1_2")(zp12)
    if poolingType == 'average':
        p1 = AveragePooling2D((2, 2), strides=(2, 2))(c12)
    else:
        p1 = MaxPooling2D((2, 2), strides=(2, 2))(c12)

    zp21 = ZeroPadding2D((1, 1))(p1)
    c21 = Convolution2D(128, 3, 3, activation='relu', trainable=trainable, name="conv_2_1")(zp21)
    zp22 = ZeroPadding2D((1, 1))(c21)
    c22 = Convolution2D(128, 3, 3, activation='relu', trainable=trainable, name="conv_2_2")(zp22)
    if poolingType == 'average':
        p2 = AveragePooling2D((2, 2), strides=(2, 2))(c22)
    else:
        p2 = MaxPooling2D((2, 2), strides=(2, 2))(c22)

    zp31 = ZeroPadding2D((1, 1))(p2)
    c31 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_1")(zp31)
    zp32 = ZeroPadding2D((1, 1))(c31)
    c32 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_2")(zp32)
    zp33 = ZeroPadding2D((1, 1))(c32)
    c33 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_3")(zp33)
    zp34 = ZeroPadding2D((1, 1))(c33)
    c34 = Convolution2D(256, 3, 3, activation='relu', trainable=trainable, name="conv_3_4")(zp34)
    if poolingType == 'average':
        p3 = AveragePooling2D((2, 2), strides=(2, 2))(c34)
    else:
        p3 = MaxPooling2D((2, 2), strides=(2, 2))(c34)
        

    zp41 = ZeroPadding2D((1, 1))(p3)
    c41 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_4_1")(zp41)
    zp42 = ZeroPadding2D((1, 1))(c41)
    c42 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_4_2")(zp42)
    zp43 = ZeroPadding2D((1, 1))(c42)
    c43 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_4_3")(zp43)
    zp44 = ZeroPadding2D((1, 1))(c43)
    c44 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_4_4")(zp44)
    if poolingType == 'average':
        p4 = AveragePooling2D((2, 2), strides=(2, 2))(c44)
    else:
        p4 = MaxPooling2D((2, 2), strides=(2, 2))(c44)

    zp51 = ZeroPadding2D((1, 1))(p4)
    c51 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_5_1")(zp51)
    zp52 = ZeroPadding2D((1, 1))(c51)
    c52 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_5_2")(zp52)
    zp53 = ZeroPadding2D((1, 1))(c52)
    c53 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_5_3")(zp53)
    zp54 = ZeroPadding2D((1, 1))(c53)
    c54 = Convolution2D(512, 3, 3, activation='relu', trainable=trainable, name="conv_5_4")(zp54)

    model = Model(input=[input], output=[
        c11, c12, 
        c21, c22, 
        c31, c32, c33, c34,
        c41, c42, c43, c44,
        c51, c52, c53, c54]
    )

    if weights_path:
        model.load_weights(weights_path)

    return model

def get_layer_data(model, pattern=''):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    if re != '':
        layers_names = [l for l in layer_dict if len(re.findall(pattern, l))]
    else:
        layers_names = [l for l in layer_dict]
    layers_names.sort()

    return (layer_dict, layers_names)
