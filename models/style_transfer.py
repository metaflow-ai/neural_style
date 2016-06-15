import os, sys
dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

from keras import backend as K
from keras.engine import merge
from keras.layers.convolutional import (Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, ELU
from keras.layers.core import Activation
from keras.layers import Input
from keras.models import Model

from models.layers.ConvolutionTranspose2D import ConvolutionTranspose2D
from models.layers.ScaledSigmoid import ScaledSigmoid

from utils.general import export_model

# inputs th ordering, BGR
def style_transfer_conv_transpose(weights_path=None, input_shape=(3, 600, 600), nb_res_layer=6):
    input = Input(shape=input_shape, name='input', dtype='float32')

    # Downsampling
    c11 = Convolution2D(32, 9, 9, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
    bn11 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c11)
    a11 = Activation('relu')(bn11)

    c12 = Convolution2D(64, 3, 3, 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(a11)
    bn12 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c12)
    a12 = Activation('relu')(bn12)

    c13 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a12)
    bn13 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c13)
    last_out = Activation('relu')(bn13)

    for i in range(nb_res_layer):
        c = Convolution2D(128, 3, 3, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(last_out)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = Activation('relu')(bn)
        c = Convolution2D(128, 3, 3, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        # a = Activation('relu')(bn)
        last_out = merge([last_out, bn], mode='sum')
        # last_out = a

    ct71 = ConvolutionTranspose2D(64, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(last_out)
    bn71 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(ct71)
    a71 = Activation('relu')(bn71)
    
    ct81 = ConvolutionTranspose2D(32, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a71)
    bn81 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(ct81)
    a81 = Activation('relu')(bn81)    

    c91 = ConvolutionTranspose2D(3, 9, 9, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a81)
    out = ScaledSigmoid(scaling=255.)(c91)    

    
    model = Model(input=[input], output=[out])

    if weights_path:
        model.load_weights(weights_path)

    return model

# Moving from 6 to 12 layers doesn't seem to improve much
def style_transfer_conv_inception(weights_path=None, input_shape=(3, 600, 600), nb_res_layer=6):
    input = Input(shape=input_shape, name='input', dtype='float32')

    # Downsampling
    c = Convolution2D(13, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(input)
    bn11 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
    a11 = Activation('relu')(bn11)
    mp11 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(input)
    m = merge([a11, mp11], mode='concat', concat_axis=1) # 16 layers

    c12 = Convolution2D(48, 3, 3, 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(m)
    bn12 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c12)
    a12 = Activation('relu')(bn12)
    mp12 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(m)
    m = merge([a12, mp12], mode='concat', concat_axis=1) # 64 layers

    c13 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(m)
    bn13 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c13)
    last_out = Activation('relu')(bn13)

    for i in range(nb_res_layer):
        #bottleneck archi
        c = Convolution2D(32, 1, 1, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(last_out)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = Activation('relu')(bn)

        # Convolutions
        c = Convolution2D(32, 3, 3, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = Activation('relu')(bn)
        c = Convolution2D(32, 3, 3, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = Activation('relu')(bn)

        #Reverse bottleneck
        c = Convolution2D(128, 1, 1, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        last_out = merge([last_out, bn], mode='sum')

    ct71 = ConvolutionTranspose2D(64, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(last_out)
    bn71 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(ct71)
    a71 = Activation('relu')(bn71)
    
    ct81 = ConvolutionTranspose2D(16, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a71)
    bn81 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(ct81)
    a81 = Activation('relu')(bn81)    

    c = Convolution2D(3, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a81)
    out = ScaledSigmoid(scaling=255.)(c)

    model = Model(input=[input], output=[out])

    if weights_path:
        model.load_weights(weights_path)

    return model

def style_transfer_conv_inception_2(weights_path=None, input_shape=(3, 600, 600), nb_res_layer=6):
    input = Input(shape=input_shape, name='input', dtype='float32')

    # Downsampling
    c = Convolution2D(13, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(input)
    bn11 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
    a11 = PReLU()(bn11) 
    mp11 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(input)
    m = merge([a11, mp11], mode='concat', concat_axis=1) # 16 layers

    c12 = Convolution2D(48, 3, 3, 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(m)
    bn12 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c12)
    a12 = PReLU()(bn12)
    mp12 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(m)
    m = merge([a12, mp12], mode='concat', concat_axis=1) # 64 layers

    c13 = Convolution2D(128, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(m)
    bn13 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c13)
    last_out = PReLU()(bn13)

    for i in range(nb_res_layer):
        #bottleneck archi
        c = Convolution2D(32, 1, 1, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(last_out)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = Activation('relu')(bn)

        # Convolutions
        c = Convolution2D(32, 3, 3, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = Activation('relu')(bn)
        c = Convolution2D(32, 3, 3, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = Activation('relu')(bn)

        #Reverse bottleneck
        c = Convolution2D(128, 1, 1, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        last_out = merge([last_out, bn], mode='sum')

    ct71 = ConvolutionTranspose2D(64, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(last_out)
    bn71 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(ct71)
    
    ct81 = ConvolutionTranspose2D(16, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(bn71)
    bn81 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(ct81)

    c = Convolution2D(3, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(bn81)
    out = ScaledSigmoid(scaling=255.)(c)

    model = Model(input=[input], output=[out])

    if weights_path:
        model.load_weights(weights_path)

    return model


def style_transfer_conv_inception_3(weights_path=None, input_shape=(3, 600, 600), nb_res_layer=6):
    input = Input(shape=input_shape, name='input', dtype='float32')

    # Downsampling
    c = Convolution2D(29, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(input)
    bn11 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
    a11 = Activation('relu')(bn11)
    mp11 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(input)
    m = merge([a11, mp11], mode='concat', concat_axis=1) # 32 layers

    c12 = Convolution2D(96, 3, 3, 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(m)
    bn12 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c12)
    a12 = Activation('relu')(bn12)
    mp12 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(m)
    m = merge([a12, mp12], mode='concat', concat_axis=1) # 64 layers

    c13 = Convolution2D(256, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(m)
    bn13 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c13)
    last_out = Activation('relu')(bn13)

    for i in range(nb_res_layer):
        #bottleneck archi
        c = Convolution2D(64, 1, 1, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(last_out)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = Activation('relu')(bn)

        # Convolutions
        c = Convolution2D(64, 3, 3, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = Activation('relu')(bn)
        c = Convolution2D(64, 3, 3, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = Activation('relu')(bn)

        #Reverse bottleneck
        c = Convolution2D(256, 1, 1, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        last_out = merge([last_out, bn], mode='sum')
        last_out = ScaledSigmoid(scaling=255.)(last_out)

    ct71 = ConvolutionTranspose2D(128, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(last_out)
    bn71 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(ct71)
    a71 = Activation('relu')(bn71)
    
    ct81 = ConvolutionTranspose2D(32, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a71)
    bn81 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(ct81)
    a81 = Activation('relu')(bn81)    

    c = Convolution2D(3, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a81)
    out = ScaledSigmoid(scaling=255.)(c)

    model = Model(input=[input], output=[out])

    if weights_path:
        model.load_weights(weights_path)

    return model

# Doesn't give beter result
def style_transfer_conv_inception_ELU_flattened(weights_path=None, input_shape=(3, 600, 600), nb_res_layer=6):
    input = Input(shape=input_shape, name='input', dtype='float32')

    # Downsampling
    c_hori = Convolution2D(13, 1, 9, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(input)
    c_verti = Convolution2D(13, 9, 1, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(input)
    m = merge([c_hori, c_verti], mode='sum')
    bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(m)
    a = ELU()(bn)
    mp = AveragePooling2D(pool_size=(2, 2), border_mode='same')(input)
    m1 = merge([a, mp], mode='concat', concat_axis=1) # 16 layers
    

    c_hori = Convolution2D(48, 1, 5, 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(m1)
    c_verti = Convolution2D(48, 5, 1, 
        init='he_normal', subsample=(2, 2),  border_mode='same', activation='linear')(m1)
    m = merge([c_hori, c_verti], mode='sum')
    bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(m)
    a = ELU()(bn)
    mp = AveragePooling2D(pool_size=(2, 2), border_mode='same')(m1)
    m = merge([a, mp], mode='concat', concat_axis=1) # 64 layers

    c = Convolution2D(128, 3, 1, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(m)
    c = Convolution2D(128, 1, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(c)
    bn13 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
    last_out = ELU()(bn13)

    for i in range(nb_res_layer):
        #bottleneck archi
        c = Convolution2D(32, 1, 1, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(last_out)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = ELU()(bn)

        # Convolutions
        c = Convolution2D(32, 1, 5, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        c = Convolution2D(32, 5, 1, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(c)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = ELU()(bn)
        c = Convolution2D(32, 1, 5, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        c = Convolution2D(32, 5, 1, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(c)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        a = ELU()(bn)

        #Reverse bottleneck
        c = Convolution2D(128, 1, 1, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a)
        bn = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(c)
        last_out = merge([last_out, bn], mode='sum')

    ct71 = ConvolutionTranspose2D(64, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(last_out)
    bn71 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(ct71)
    a71 = ELU()(bn71)
    
    ct81 = ConvolutionTranspose2D(16, 3, 3, 
        init='he_normal', subsample=(2, 2), border_mode='same', activation='linear')(a71)
    bn81 = BatchNormalization(axis=1, momentum=0.1, gamma_init='he_normal')(ct81)
    a81 = ELU()(bn81)    

    c = Convolution2D(3, 3, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(a81)
    c = Convolution2D(3, 3, 1, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(c)
    out = Convolution2D(3, 1, 3, 
        init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(c)
    out = ScaledSigmoid(scaling=255.)(out)

    model = Model(input=[input], output=[out])

    if weights_path:
        model.load_weights(weights_path)

    return model
    

if __name__ == "__main__":
    from keras.utils.visualize_util import plot as plot_model

    dir = os.path.dirname(os.path.realpath(__file__))
    resultsDir = dir + '/data/st'

    model = style_transfer_conv_transpose()
    export_model(model, resultsDir + '/style_transfer_conv_transpose')
    plot_model(model, resultsDir + '/style_transfer_conv_transpose/model.png', True)

    model = style_transfer_conv_inception()
    export_model(model, resultsDir + '/style_transfer_conv_inception')
    plot_model(model, resultsDir + '/style_transfer_conv_inception/model.png', True)

    model = style_transfer_conv_inception_2()
    export_model(model, resultsDir + '/style_transfer_conv_inception_2')
    plot_model(model, resultsDir + '/style_transfer_conv_inception_2/model.png', True)

    model = style_transfer_conv_inception_3()
    export_model(model, resultsDir + '/style_transfer_conv_inception_3')
    plot_model(model, resultsDir + '/style_transfer_conv_inception_3/model.png', True)

    # model = style_transfer_conv_inception_ELU_flattened()
    # export_model(model, resultsDir + '/style_transfer_conv_inception_ELU_flattened')
    # plot_model(model, resultsDir + '/style_transfer_conv_inception_ELU_flattened/model.png', True)
