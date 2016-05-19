import os, sys

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

from utils.copy_seq_weights import copySeqWeights
from model_headless import *

copySeqWeights(VGG_16_headless_5(), 
    dir + '/vgg-16_weights.hdf5', 
    dir + '/vgg-16_headless_5_weights.hdf5'
)

copySeqWeights(VGG_16_headless_4(), 
    dir + '/vgg-16_weights.hdf5', 
    dir + '/vgg-16_headless_4_weights.hdf5'
)

copySeqWeights(VGG_16_headless_3(), 
    dir + '/vgg-16_weights.hdf5', 
    dir + '/vgg-16_headless_3_weights.hdf5'
)

copySeqWeights(VGG_16_headless_2(), 
    dir + '/vgg-16_weights.hdf5', 
    dir + '/vgg-16_headless_2_weights.hdf5'
)

copySeqWeights(VGG_16_headless_1(), 
    dir + '/vgg-16_weights.hdf5', 
    dir + '/vgg-16_headless_1_weights.hdf5'
)
