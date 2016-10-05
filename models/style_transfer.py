import os, sys
dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

from keras import backend as K
from keras.engine import merge
from keras.layers.convolutional import (Convolution2D, MaxPooling2D, AveragePooling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.core import Activation
from keras.layers import Input
from keras.models import Model

from models.layers.ATrousConvolution2D import ATrousConvolution2D
from models.layers.ConvolutionTranspose2D import ConvolutionTranspose2D
from models.layers.ScaledSigmoid import ScaledSigmoid
from models.layers.PhaseShift import PhaseShift
from models.layers.InstanceNormalization import InstanceNormalization

from utils.general import export_model

# inputs th ordering, BGR
def st_convt(input_shape, weights_path=None, mode=0, nb_res_layer=4):
    if K.image_dim_ordering() == 'tf':
        channel_axis = 3
    else:
        channel_axis = 1

    input = Input(shape=input_shape, name='input_node', dtype=K.floatx())
    # Downsampling
    c11 = Convolution2D(32, 9, 9, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    bn11 = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(c11)
    a11 = Activation('relu')(bn11)

    c12 = Convolution2D(64, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(a11)
    bn12 = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(c12)
    a12 = Activation('relu')(bn12)

    c13 = Convolution2D(128, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a12)
    bn13 = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(c13)
    last_out = Activation('relu')(bn13)

    for i in range(nb_res_layer):
        c = Convolution2D(128, 3, 3, dim_ordering=K.image_dim_ordering(), 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(last_out)
        bn = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(c)
        a = Activation('relu')(bn)
        c = Convolution2D(128, 3, 3, dim_ordering=K.image_dim_ordering(), 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(c)
        # a = Activation('relu')(bn)
        last_out = merge([last_out, bn], mode='sum')
        # last_out = a

    ct71 = ConvolutionTranspose2D(64, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(last_out)
    bn71 = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(ct71)
    a71 = Activation('relu')(bn71)
    
    ct81 = ConvolutionTranspose2D(32, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a71)
    bn81 = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(ct81)
    a81 = Activation('relu')(bn81)    

    c91 = Convolution2D(3, 9, 9, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a81)
    out = ScaledSigmoid(scaling=255., name="output_node")(c91)    

    
    model = Model(input=[input], output=[out])

    if weights_path:
        model.load_weights(weights_path)

    return model

# Moving from 4 to 12 layers doesn't seem to improve much
def st_conv_inception(input_shape, weights_path=None, mode=0, nb_res_layer=4):
    if K.image_dim_ordering() == 'tf':
        channel_axis = 3
    else:
        channel_axis = 1

    input = Input(shape=input_shape, name='input_node', dtype=K.floatx())
    # Downsampling
    c = Convolution2D(13, 9, 9, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(input)
    bn11 = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(c)
    a11 = Activation('relu')(bn11)
    mp11 = MaxPooling2D(pool_size=(2, 2), dim_ordering=K.image_dim_ordering(), border_mode='same')(input)
    m = merge([a11, mp11], mode='concat', concat_axis=channel_axis) # 16 layers

    c12 = Convolution2D(48, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(m)
    bn12 = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(c12)
    a12 = Activation('relu')(bn12)
    mp12 = MaxPooling2D(pool_size=(2, 2), dim_ordering=K.image_dim_ordering(), border_mode='same')(m)
    m = merge([a12, mp12], mode='concat', concat_axis=channel_axis) # 64 layers

    c13 = Convolution2D(128, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(m)
    bn13 = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(c13)
    last_out = Activation('relu')(bn13)

    for i in range(nb_res_layer):
        out = naive_inception_layer(last_out, K.image_dim_ordering(), channel_axis, mode)
        last_out = merge([last_out, out], mode='sum')

    ct = ConvolutionTranspose2D(64, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(last_out)
    bn = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(ct)
    a = Activation('relu')(bn)
    
    ct = ConvolutionTranspose2D(16, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a)
    bn = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(ct)
    a = Activation('relu')(bn)    

    c = Convolution2D(3, 9, 9, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
    out = ScaledSigmoid(scaling=255., name="output_node")(c)

    model = Model(input=[input], output=[out])

    if weights_path:
        model.load_weights(weights_path)

    return model

def st_convt_inception_prelu(input_shape, weights_path=None, mode=0, nb_res_layer=4):
    if K.image_dim_ordering() == 'tf':
        channel_axis = 3
    else:
        channel_axis = 1

    input = Input(shape=input_shape, name='input_node', dtype=K.floatx())
    # Downsampling
    c = Convolution2D(13, 9, 9, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(input)
    bn11 = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(c)
    a11 = PReLU()(bn11) 
    mp11 = MaxPooling2D(pool_size=(2, 2), dim_ordering=K.image_dim_ordering(), border_mode='same')(input)
    m = merge([a11, mp11], mode='concat', concat_axis=channel_axis) # 16 layers

    c12 = Convolution2D(48, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(m)
    bn12 = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(c12)
    a12 = PReLU()(bn12)
    mp12 = MaxPooling2D(pool_size=(2, 2), dim_ordering=K.image_dim_ordering(), border_mode='same')(m)
    m = merge([a12, mp12], mode='concat', concat_axis=channel_axis) # 64 layers

    c13 = Convolution2D(128, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(m)
    out = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(c13)
    last_out = PReLU()(out)

    for i in range(nb_res_layer):
        out = naive_inception_layer(last_out, K.image_dim_ordering(), channel_axis, mode, activation_type='prelu')
        last_out = merge([last_out, out], mode='sum')

    ct = ConvolutionTranspose2D(64, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(last_out)
    bn = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(ct)
    a = PReLU()(bn)
    
    ct = ConvolutionTranspose2D(16, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a)
    bn = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.1, gamma_init='he_normal')(ct)
    a = PReLU()(bn)    

    c = Convolution2D(3, 9, 9, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
    out = ScaledSigmoid(scaling=255., name="output_node")(c)

    model = Model(input=[input], output=[out])

    if weights_path:
        model.load_weights(weights_path)

    return model


def st_conv_inception_4(input_shape, weights_path=None, mode=0, nb_res_layer=4):
    if K.image_dim_ordering() == 'tf':
        channel_axis = 3
    else:
        channel_axis = 1

    input = Input(shape=input_shape, name='input_node', dtype=K.floatx())
    # Downsampling
    c = Convolution2D(13, 7, 7, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(input)
    out = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(c)
    a = PReLU()(out) 
    p = AveragePooling2D(pool_size=(2, 2), dim_ordering=K.image_dim_ordering(), border_mode='same')(input)
    m = merge([a, p], mode='concat', concat_axis=channel_axis) # 16 layers

    c = Convolution2D(48, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(m)
    a = PReLU()(c)
    p = AveragePooling2D(pool_size=(2, 2), dim_ordering=K.image_dim_ordering(), border_mode='same')(m)
    m = merge([a, p], mode='concat', concat_axis=channel_axis) # 64 layers

    out = Convolution2D(128, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(m)
    last_out = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(out)
    last_out = Activation('relu')(last_out)

    for i in range(nb_res_layer):
        out = inception_layer(last_out, K.image_dim_ordering(), channel_axis, mode)
        last_out = merge([last_out, out], mode='sum')

    out = Convolution2D(128, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(last_out)
    out = ConvolutionTranspose2D(3, 5, 5, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(4, 4), border_mode='same', activation='linear')(out)
    out = ScaledSigmoid(scaling=255., name="output_node")(out)

    model = Model(input=[input], output=[out])

    if weights_path:
        model.load_weights(weights_path)

    return model

def st_conv_inception_4_superresolution(input_shape, weights_path=None, mode=0, nb_res_layer=4):
    if K.image_dim_ordering() == 'tf':
        channel_axis = 3
    else:
        channel_axis = 1

    input = Input(shape=input_shape, name='input_node', dtype=K.floatx())
    out = Convolution2D(128, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    last_out = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(out)
    last_out = Activation('relu')(last_out)

    for i in range(nb_res_layer):
        out = inception_layer(last_out, K.image_dim_ordering(), channel_axis, mode)
        last_out = merge([last_out, out], mode='sum')

    out = Convolution2D(128, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(last_out)
    out = ConvolutionTranspose2D(3, 5, 5, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(4, 4), border_mode='same', activation='linear')(out)
    out = ScaledSigmoid(scaling=255., name="output_node")(out)

    model = Model(input=[input], output=[out])

    if weights_path:
        model.load_weights(weights_path)

    return model

def st_conv_inception_4_fast(input_shape, weights_path=None, mode=0, nb_res_layer=4):
    if K.image_dim_ordering() == 'tf':
        channel_axis = 3
    else:
        channel_axis = 1

    input = Input(shape=input_shape, name='input_node', dtype=K.floatx())
    # Downsampling
    c = Convolution2D(13, 7, 7, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(input)
    out = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.5, gamma_init='he_normal')(c)
    a = Activation('relu')(out) 
    p = AveragePooling2D(pool_size=(2, 2), dim_ordering=K.image_dim_ordering(), border_mode='same')(input)
    m = merge([a, p], mode='concat', concat_axis=channel_axis) # 16 layers

    c = Convolution2D(64, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(m)
    out = BatchNormalization(mode=mode, axis=channel_axis, momentum=0.5, gamma_init='he_normal')(c)
    last_out = Activation('relu')(out)
    # p = AveragePooling2D(pool_size=(2, 2), dim_ordering=K.image_dim_ordering(), border_mode='same')(m)
    # last_out = merge([a, p], mode='concat', concat_axis=channel_axis) # 64 layers


    for i in range(nb_res_layer):
        out = inception_layer_fast(last_out, K.image_dim_ordering(), channel_axis, mode, 64)
        last_out = merge([last_out, out], mode='sum')

    out = Convolution2D(64, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(last_out)
    out = ConvolutionTranspose2D(16, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(out)
    out = ConvolutionTranspose2D(3, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(out)
    out = ScaledSigmoid(scaling=255., name="output_node")(out)

    model = Model(input=[input], output=[out])

    if weights_path:
        model.load_weights(weights_path)

    return model
    
def naive_inception_layer(input, do, channel_axis, batchnorm_mode, activation_type='relu'):
    # Bottleneck
    out = Convolution2D(32, 1, 1, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    out = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(out)
    if activation_type == 'prelu':
        out = PReLU()(out)
    else:
        out = Activation('relu')(out)

    # Convolutions
    out = Convolution2D(32, 3, 3, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(out)
    if activation_type == 'prelu':
        out = PReLU()(out)
    else:
        out = Activation('relu')(out)
    out = Convolution2D(32, 3, 3, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(out)
    if activation_type == 'prelu':
        out = PReLU()(out)
    else:
        out = Activation('relu')(out)

    # Reverse bottleneck
    out = Convolution2D(128, 1, 1, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(out)

    return out

def inception_layer(input, do, channel_axis, batchnorm_mode):
    # Branch 1
    out = Convolution2D(32, 1, 1, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    out = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(out)
    out1 = Activation('relu')(out)

    # Branch 2
    out = Convolution2D(32, 1, 1, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    out = Activation('relu')(out)
    out = Convolution2D(32, 3, 3, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(out)
    out2 = Activation('relu')(out)

    # Branch 3
    out = Convolution2D(32, 1, 1, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    out = Activation('relu')(out)
    out = Convolution2D(32, 5, 5, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(out)
    out3 = Activation('relu')(out)

    # Branch 4
    out = Convolution2D(32, 1, 1, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    out = Activation('relu')(out)
    out = Convolution2D(32, 3, 3, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = Activation('relu')(out)
    out = Convolution2D(32, 3, 3, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.9, gamma_init='he_normal')(out)
    out4 = Activation('relu')(out)

    m = merge([out1, out2, out3, out4], mode='concat', concat_axis=channel_axis) # 16 layers

    return m

def inception_layer_fast(input, do, channel_axis, batchnorm_mode, nb_layers):
    # Branch 1
    out = Convolution2D(int(nb_layers/4), 1, 1, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    out = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.5, gamma_init='he_normal')(out)
    out1 = Activation('relu')(out)

    # Branch 2
    out = Convolution2D(int(nb_layers/4), 1, 1, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    out = Convolution2D(int(nb_layers/4), 3, 3, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.5, gamma_init='he_normal')(out)
    out2 = Activation('relu')(out)

    # Branch 3
    out = Convolution2D(int(nb_layers/4), 1, 1, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    out = Convolution2D(int(nb_layers/4), 3, 3, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = Convolution2D(int(nb_layers/4), 3, 3, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.5, gamma_init='he_normal')(out)
    out3 = Activation('relu')(out)

    # Branch 4
    out = Convolution2D(int(nb_layers/4), 1, 1, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    out = Convolution2D(int(nb_layers/4), 3, 3, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = Convolution2D(int(nb_layers/4), 3, 3, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = Convolution2D(int(nb_layers/4), 3, 3, dim_ordering=do, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(out)
    out = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.5, gamma_init='he_normal')(out)
    out4 = Activation('relu')(out)

    m = merge([out1, out2, out3, out4], mode='concat', concat_axis=channel_axis) # 16 layers
    m = BatchNormalization(mode=batchnorm_mode, axis=channel_axis, momentum=0.5, gamma_init='he_normal')(m)

    return m

def fast_st_ps(input_shape, weights_path=None, mode=0, nb_res_layer=4):
    input = Input(shape=input_shape, name='input_node', dtype=K.floatx())
    # Downsampling
    c11 = Convolution2D(32, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    bn11 = InstanceNormalization('inorm-1')(c11)
    a11 = Activation('relu')(bn11)

    c12 = Convolution2D(64, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(a11)
    bn12 = InstanceNormalization('inorm-2')(c12)
    a12 = Activation('relu')(bn12)

    c13 = Convolution2D(128, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a12)
    bn13 = InstanceNormalization('inorm-3')(c13)
    last_out = Activation('relu')(bn13)

    for i in range(nb_res_layer):
        c = Convolution2D(128, 3, 3, dim_ordering=K.image_dim_ordering(), 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(last_out)
        bn = InstanceNormalization('inorm-res-%d' % i)(c)
        a = Activation('relu')(bn)
        c = Convolution2D(128, 3, 3, dim_ordering=K.image_dim_ordering(), 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = InstanceNormalization('inorm-5-%d' % i)(c)
        # a = Activation('relu')(bn)
        last_out = merge([last_out, bn], mode='sum')
        # last_out = a

    ct71 = Convolution2D(48, 3, 3, dim_ordering=K.image_dim_ordering(), 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(last_out)
    out = PhaseShift(ratio=4, color=True)(ct71)
    out = ScaledSigmoid(scaling=255., name="output_node")(out)    

    
    model = Model(input=[input], output=[out])

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    from keras.utils.visualize_util import plot as plot_model

    if K._BACKEND == "tensorflow":
        K.set_image_dim_ordering('tf')
    else:
        K.set_image_dim_ordering('th')

    if K.image_dim_ordering() == 'th':
        input_shape = (3, 600, 600)
    else:
        input_shape = (600, 600, 3)

    dir = os.path.dirname(os.path.realpath(__file__))
    results_dir = dir + '/data/st'

    # print('exporting st_convt')
    # model = st_convt(input_shape=input_shape, nb_res_layer=2)
    # export_model(model, results_dir + '/st_convt')
    # plot_model(model, results_dir + '/st_convt/model.png', True)

    # print('exporting st_conv_inception')
    # model = st_conv_inception(input_shape=input_shape, nb_res_layer=2)
    # export_model(model, results_dir + '/st_conv_inception')
    # plot_model(model, results_dir + '/st_conv_inception/model.png', True)

    # print('exporting st_convt_inception_prelu')
    # model = st_convt_inception_prelu(input_shape=input_shape, nb_res_layer=2)
    # export_model(model, results_dir + '/st_convt_inception_prelu')
    # plot_model(model, results_dir + '/st_convt_inception_prelu/model.png', True)

    # print('exporting st_conv_inception_4')
    # model = st_conv_inception_4(input_shape=input_shape, nb_res_layer=2)
    # export_model(model, results_dir + '/st_conv_inception_4')
    # plot_model(model, results_dir + '/st_conv_inception_4/model.png', True)

    # print('exporting st_conv_inception_4_fast')
    # model = st_conv_inception_4_fast(input_shape=input_shape, nb_res_layer=4)
    # export_model(model, results_dir + '/st_conv_inception_4_fast')
    # plot_model(model, results_dir + '/st_conv_inception_4_fast/model.png', True)

    print('exporting fast_st_ps')
    model = fast_st_ps(input_shape=input_shape, nb_res_layer=4)
    export_model(model, results_dir + '/fast_st_ps')
    plot_model(model, results_dir + '/fast_st_ps/model.png', True)

    # print('exporting st_conv_inception_4_superresolution')
    # if K.image_dim_ordering() == 'th':
    #     input_shape = (3, int(600/4), int(600/4))
    # else:
    #     input_shape = (int(600/4), int(600/4), 3)
    # model = st_conv_inception_4_superresolution(input_shape=input_shape, nb_res_layer=2)
    # export_model(model, results_dir + '/st_conv_inception_4_superresolution')
    # plot_model(model, results_dir + '/st_conv_inception_4_superresolution/model.png', True)
