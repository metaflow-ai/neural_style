import numpy as np
import os, time
import png

from models.style_transfer import style_transfer

from vgg16.model import VGG_16_mean 
from utils.imutils import *
from utils.lossutils import *

dir = os.path.dirname(os.path.realpath(__file__))
vgg16Dir = dir + '/vgg16'
dataDir = dir + '/data'
outputDir = dataDir + '/output'
if not os.path.isdir(outputDir): 
    os.makedirs(outputDir)
testDir = dataDir + '/test'

channels = 3
width = 600
height = 600
input_shape = (channels, width, height)
batch = 4

print('Loading test set')
# X_test = load_images(testDir, size=(height, width))
X_test = load_images(dir + '/data/overfit', limit=40, size=(height, width))
print('X_test.shape: ' + str(X_test.shape))

print('Loading mean')
meanPath = vgg16Dir + '/vgg-16_mean.npy'
mean = VGG_16_mean(path=meanPath)
print("mean shape: " + str(mean.shape))

print('Loading style_transfer')
stWeights = dir + '/models/results/st/st_vangogh_weights.hdf5'
st_model = style_transfer(stWeights)

predict = K.function([st_model.input, K.learning_phase()], st_model.output)

print('Predicting')
# time it
start= time.clock()
num_loop = 1
for i in range(num_loop):
    # results = st_model.predict(X_test) # Equivalent to predict([X_test, False])
    results = predict([X_test, True])
end= time.clock()
time= (end-start)/(X_test.shape[0] * num_loop)
print("time taken on 1 average call: " + str(time))


print('Dumping results')
for idx, im in enumerate(results):
    prefix = str(idx).zfill(4)
    fullOutPath = outputDir + '/' + prefix + ".png"
    deprocess_image(fullOutPath, im)

    fullOriPath = outputDir + '/' + prefix + "_ori.png"
    deprocess_image(fullOriPath, X_test[idx], False)

