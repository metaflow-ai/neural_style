import os, sys

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

import numpy as np
import h5py, gc

from keras import backend as K

from vgg19.model_headless import *

from utils.imutils import *
from utils.lossutils import *

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/../vgg19'
modelWeights = vgg19Dir + '/vgg-19_headless_5_weights.hdf5'
paintingsDir = dir + '/paintings'

model = VGG_19_headless_5(modelWeights, trainable=False, poolingType='average')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
input_layer = model.input
layers_used = ['conv_1_1', 'conv_1_2', 'conv_2_1', 'conv_2_2', 'conv_3_1', 'conv_3_4', 'conv_4_1', 'conv_4_4', 'conv_5_1', 'conv_5_4']
outputs_layer = [grams(layer_dict[name].output) for name in layers_used]

predict = K.function([input_layer], outputs_layer)

print('Loading train images')
paintings_fullpath = dir + '/paintings'
filenames = [f for f in os.listdir(paintings_fullpath) if len(re.findall('\.(jpg|png)$', f))]
for filename in filenames:
    print('Loading: ' + paintings_fullpath + '/' + filename)
    painting = np.array([load_image(paintings_fullpath + '/' + filename, size=None, dim_ordering='th')])
    print("painting shape: " + str(painting.shape))

    print('Creating training labels')
    painting_label = predict([painting])

    print('Saving data')
    f = h5py.File(paintings_fullpath + "/" + filename.split('.')[0] + '_ori.hdf5', "w")
    for idx, layer_name in enumerate(layers_used):
        print(painting_label[idx].shape)
        f.create_dataset(layer_name, data=painting_label[idx])
    f.close()

    gc.collect()

    print('Loading: ' + paintings_fullpath + '/' + filename)
    painting = np.array([load_image(paintings_fullpath + '/' + filename, size=(600, 600), dim_ordering='th')])
    print("painting shape: " + str(painting.shape))

    print('Creating training labels')
    painting_label = predict([painting])

    print('Saving data')
    f = h5py.File(paintings_fullpath + "/" + filename.split('.')[0] + '_ori.hdf5', "w")
    for idx, layer_name in enumerate(layers_used):
        print(painting_label[idx].shape)
        f.create_dataset(layer_name, data=painting_label[idx])
    f.close()

    gc.collect()

    print('Loading: ' + paintings_fullpath + '/' + filename)
    painting = np.array([load_image(paintings_fullpath + '/' + filename, size=(256, 256), dim_ordering='th')])
    print("painting shape: " + str(painting.shape))

    print('Creating training labels')
    painting_label = predict([painting])

    print('Saving data')
    f = h5py.File(paintings_fullpath + "/" + filename.split('.')[0] + '_ori.hdf5', "w")
    for idx, layer_name in enumerate(layers_used):
        print(painting_label[idx].shape)
        f.create_dataset(layer_name, data=painting_label[idx])
    f.close()

    gc.collect()