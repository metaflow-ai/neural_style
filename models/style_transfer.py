from keras import backend as K
from keras.engine import merge
from keras.layers.convolutional import (Convolution2D, UpSampling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Input
from keras.models import Model

from models.layers.ConvolutionTranspose2D import ConvolutionTranspose2D
from models.layers.ScaledSigmoid import ScaledSigmoid

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
    