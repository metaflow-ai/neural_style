import os, sys

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

from keras import backend as K
from utils.general import copySeqWeights
from model_headless import VGG_19_headless_5


K.set_image_dim_ordering('tf')
copySeqWeights(VGG_19_headless_5(input_shape=(256, 256, 3)), 
    dir + '/vgg-19_weights.hdf5', 
    dir + '/vgg-19-' + K.image_dim_ordering() + '-' + K._BACKEND + '_headless_5_weights.hdf5'
)

K.set_image_dim_ordering('th')
copySeqWeights(VGG_19_headless_5(input_shape=(3, 256, 256)), 
    dir + '/vgg-19_weights.hdf5', 
    dir + '/vgg-19-' + K.image_dim_ordering() + '-' + K._BACKEND + '_headless_5_weights.hdf5'
)

