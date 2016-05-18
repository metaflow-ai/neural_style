import os, sys

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

import numpy as np
import h5py, gc

from keras import backend as K

from vgg16.model import VGG_16_mean 
from vgg16.model_headless import *

from utils.imutils import *
from utils.lossutils import *

vgg16Dir = dir + '/../vgg16'
modelWeights = vgg16Dir + '/vgg-16_headless_5_weights.hdf5'
paintingsDir = dir + '/paintings'

print('Loading mean')
meanPath = vgg16Dir + '/vgg-16_mean.npy'
mean = VGG_16_mean(path=meanPath)

print('Loading train images')
paintings_fullpath = dir + '/paintings'
filenames = [f for f in os.listdir(paintings_fullpath) if len(re.findall('\.(jpg|png)$', f))]
for filename in filenames:
    print(paintings_fullpath + '/' + filename)
    painting = load_image(paintings_fullpath + '/' + filename)
    print("painting shape: " + str(painting.shape))

    input_shape = painting.shape

    print('Loading VGG headless 5 model')    
    vgg_model = VGG_16_headless_5(modelWeights, input_shape=input_shape, trainable=False)
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
    input_layer = vgg_model.input
    layers_used = ['conv_1_2', 'conv_2_2', 'conv_3_3', 'conv_4_3', 'conv_5_3']
    outputs_layer = [layer_dict[name].output for name in layers_used]

    print('Creating training labels')
    predict = K.function([input_layer], outputs_layer)
    painting_label = predict([painting - mean])
    painting_label = map(lambda X: grams(X).eval(), painting_label)

    print('Saving data')
    f = h5py.File(paintings_fullpath + "/" + filename.split('.')[0] + '.hdf5', "w")
    for idx, layer_name in enumerate(layers_used):
        f.create_dataset(layer_name, data=painting_label[idx])
    f.close()

    gc.collect()
