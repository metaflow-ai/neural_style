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
meanPath = vgg16Dir + '/vgg-16_mean.npy'

print('Loading VGG headless 5')
mean = VGG_16_mean(path=meanPath)

model = VGG_16_headless_5(modelWeights, trainable=False, poolingType='average')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
input_layer = model.input
layers_used = ['conv_1_2', 'conv_2_2', 'conv_3_3', 'conv_4_3', 'conv_5_3']
outputs_layer = [layer_dict[name].output for name in layers_used]

predict = K.function([input_layer], outputs_layer)

print('Loading train images')
paintings_fullpath = dir + '/paintings'
filenames = [f for f in os.listdir(paintings_fullpath) if len(re.findall('\.(jpg|png)$', f))]
for filename in filenames:
    print(paintings_fullpath + '/' + filename)
    painting = np.array([load_image(paintings_fullpath + '/' + filename)])
    print("painting shape: " + str(painting.shape))

    print('Creating training labels')
    painting_label = predict([painting - mean])
    painting_label = map(lambda X: grams(X).eval(), painting_label)

    print('Saving data')
    f = h5py.File(paintings_fullpath + "/" + filename.split('.')[0] + '_ori.hdf5', "w")
    for idx, layer_name in enumerate(layers_used):
        f.create_dataset(layer_name, data=painting_label[idx])
    f.close()

    gc.collect()

    painting = np.array([load_image(paintings_fullpath + '/' + filename, size=(600, 600))])
    print("painting shape: " + str(painting.shape))

    print('Creating training labels')
    painting_label = predict([painting - mean])
    painting_label = map(lambda X: grams(X).eval(), painting_label)

    print('Saving data')
    f = h5py.File(paintings_fullpath + "/" + filename.split('.')[0] + '_600x600.hdf5', "w")
    for idx, layer_name in enumerate(layers_used):
        f.create_dataset(layer_name, data=painting_label[idx])
    f.close()

    gc.collect()

    painting = np.array([load_image(paintings_fullpath + '/' + filename, size=(256, 256))])
    print("painting shape: " + str(painting.shape))

    print('Creating training labels')
    painting_label = predict([painting - mean])
    painting_label = map(lambda X: grams(X).eval(), painting_label)

    print('Saving data')
    f = h5py.File(paintings_fullpath + "/" + filename.split('.')[0] + '_256x256.hdf5', "w")
    for idx, layer_name in enumerate(layers_used):
        f.create_dataset(layer_name, data=painting_label[idx])
    f.close()

    gc.collect()