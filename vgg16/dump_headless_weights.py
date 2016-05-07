import os
from utils.copy_seq_weights import copySeqWeights
from model_headless import *


dir = os.path.dirname(os.path.realpath(__file__))

model = VGG_16_headless_5()
copySeqWeights(model, 
    dir + '/vgg-16_weights.hdf5', 
    dir + '/vgg-16_headless_5_weights.hdf5'
)

model = VGG_16_headless_4()
copySeqWeights(model, 
    dir + '/vgg-16_weights.hdf5', 
    dir + '/vgg-16_headless_4_weights.hdf5'
)

model = VGG_16_headless_3()
copySeqWeights(model, 
    dir + '/vgg-16_weights.hdf5', 
    dir + '/vgg-16_headless_3_weights.hdf5'
)

model = VGG_16_headless_2()
copySeqWeights(model, 
    dir + '/vgg-16_weights.hdf5', 
    dir + '/vgg-16_headless_2_weights.hdf5'
)

model = VGG_16_headless_1()
copySeqWeights(model, 
    dir + '/vgg-16_weights.hdf5', 
    dir + '/vgg-16_headless_1_weights.hdf5'
)
