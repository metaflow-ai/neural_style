import numpy as np
import os, time
import png

from models.style_transfer import *

from vgg19.model import VGG_19_mean 
from utils.imutils import *
from utils.lossutils import *

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
resultsDir = dir + '/models/results/st2'
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
X_test = load_images(testDir, limit=4, size=(height, width))
print('X_test.shape: ' + str(X_test.shape))

print('Loading mean')
meanPath = vgg19Dir + '/vgg-16_mean.npy'
mean = VGG_19_mean(path=meanPath)
print("mean shape: " + str(mean.shape))

weights_filenames = [f for f in os.listdir(resultsDir) if len(re.findall('.*st.*weights.*\.hdf5$', f))]
current_iter = 0
for weights_filename in weights_filenames:
    fullpath = resultsDir + '/' + weights_filename
    print('Loading style_transfer with weights file: ' + fullpath)
    st_model = style_transfer(fullpath)
    predict = K.function([st_model.input, K.learning_phase()], st_model.output)

    print('Predicting')
    # results = st_model.predict(X_test) # Equivalent to predict([X_test, False])
    results = predict([X_test, True])

    print('Dumping results')
    for idx, im in enumerate(results):
        prefix = str(current_iter).zfill(4)
        fullOutPath = outputDir + '/' + prefix + "_" + str(idx) + ".png"
        deprocess(fullOutPath, im)

        # fullOriPath = outputDir + '/' + prefix + "_ori.png"
        # deprocess(fullOriPath, X_test[idx], False)

        current_iter += 1