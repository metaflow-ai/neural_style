import os, sys, time

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

import numpy as np
from keras.utils.visualize_util import plot as plot_model
from models.style_transfer import *

from vgg16.model import VGG_16_mean 
from utils.imutils import *
from utils.lossutils import *

dir = os.path.dirname(os.path.realpath(__file__))
vgg16Dir = dir + '/../vgg16'
resultsDir = dir + '/../models/results/st2'
dataDir = dir + '/../data'
testDir = dataDir + '/test'

channels = 3
width = 600
height = 600
input_shape = (channels, width, height)
batch = 4

print('Loading test set')
X_test = load_images(testDir, limit=10, size=(height, width))
print('X_test.shape: ' + str(X_test.shape))


model = style_transfer_3_3_only_double_stride_nobatchnorm()
plot_model(model, to_file=dir + '/model.png', show_shapes=True)
total_params = 0
for i in range(len(model.layers)):
    total_params += model.layers[i].count_params()
print('Total number of params:' + str(total_params))

print('Predicting')
# time it
start = time.clock()
num_loop = 1
for i in range(num_loop):
    print('loop: ' + str(i))
    results = model.predict(X_test) # Equivalent to predict([X_test, False])
end = time.clock()
duration = (end-start)/(X_test.shape[0] * num_loop)
print("duration taken on 1 average call: " + str(duration))