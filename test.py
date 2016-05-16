import numpy as np
import os, time
import png

from models.style_transfer import style_transfer

from utils.imutils import *
from utils.lossutils import *

dir = os.path.dirname(os.path.realpath(__file__))
dataDir = dir + '/data'
outputDir = dataDir + '/output'
if not os.path.isdir(outputDir): 
    os.makedirs(outputDir)
testDir = dataDir + '/test'

channels = 3
width = 256
height = 256
input_shape = (channels, width, height)
batch = 4

print('Loading test set')
# X_test = load_images(testDir, size=(height, width))
X_test = load_images(dir + '/data/overfit', size=(height, width))
print(X_test.shape)

print('Loading style_transfer')
stWeights = dir + '/models/results/st/st_vangogh_weights.hdf5'
st_model = style_transfer(stWeights)

print('Compiling st_model')
st_model.compile(loss='mean_squared_error', optimizer='sgd')

print('Predicting')
# time it
start= time.clock()
for i in range(1):
    results = st_model.predict(X_test)
end= time.clock()
time= (end-start)/1
print("time taken on 1 average call:", time)


print('Dumping results')
for idx, im in enumerate(results):
    prefix = str(idx).zfill(4)
    fullOutPath = outputDir + '/' + prefix + ".png"
    deprocess_image(fullOutPath, im)
    fullOutPath = outputDir + '/' + prefix + "_imsave.png"
    deprocess_image(fullOutPath, im, False)

    fullOriPath = outputDir + '/' + prefix + "_ori.png"
    deprocess_image(fullOriPath, X_test[idx], False)

