import os, sys

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

from utils.general import copySeqWeights
from model_headless import VGG_19_headless_5

copySeqWeights(VGG_19_headless_5(), 
    dir + '/vgg-19_weights.hdf5', 
    dir + '/vgg-19_headless_5_weights.hdf5'
)

