import os, re

from keras import backend as K

from models.style_transfer import (style_transfer_conv_transpose, style_transfer_upsample)
from utils.imutils import load_images, deprocess, save_image

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
resultsDir = dir + '/models/results/st'
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
X_test = load_images(testDir, limit=4, size=(height, width), dim_ordering='th', verbose=True)
print('X_test.shape: ' + str(X_test.shape))


weights_filenames = [f for f in os.listdir(resultsDir) if len(re.findall('.*st.*weights.*\.hdf5$', f))]
current_iter = 0
for weights_filename in weights_filenames:
    fullpath = resultsDir + '/' + weights_filename
    print('Loading style_transfer with weights file: ' + fullpath)
    st_model = style_transfer_upsample(fullpath)
    predict = K.function([st_model.input, K.learning_phase()], st_model.output)

    print('Predicting')
    # results_false = st_model.predict(X_test) # Equivalent to predict([X_test, False])
    results = predict([X_test, True])

    print('Dumping results')
    for idx, im in enumerate(results):
        prefix = str(current_iter).zfill(4)
        fullOutPath = outputDir + '/' + prefix + "_" + str(idx) + ".png"
        save_image(fullOutPath, deprocess(im, dim_ordering='th'))

        # fullFalsePath = outputDir + '/' + prefix + "_false.png"
        # save_image(fullFalsePath, deprocess(results_false[idx], dim_ordering='th'))

        current_iter += 1
