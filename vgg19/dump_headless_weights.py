import os, sys

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

from utils.copy_seq_weights import copySeqWeights
from model_headless import *

copySeqWeights(VGG_19_headless_5(), 
    dir + '/vgg-19_weights.hdf5', 
    dir + '/vgg-19_headless_5_weights.hdf5'
)

