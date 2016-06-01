import os, re

from keras import backend as K

from models.style_transfer import (style_transfer_conv_transpose)
from utils.imutils import load_images, save_image_st

dir = os.path.dirname(os.path.realpath(__file__))
dataDir = dir + '/data'
weightsDir = dir + '/models/weights/st'
if not os.path.isdir(weightsDir): 
    os.makedirs(weightsDir)
outputDir = dataDir + '/output'
if not os.path.isdir(outputDir): 
    os.makedirs(outputDir)

overfitDir = dataDir + '/overfit'    
testDir = dataDir + '/test'

channels = 3
width = 256
height = 256
input_shape = (channels, width, height)

X_overfit = load_images(overfitDir, size=(height, width), dim_ordering='th', verbose=True, st=True)
X_test = load_images(testDir, limit=20, size=(height, width), dim_ordering='th', verbose=True, st=True)
print('X_test.shape: ' + str(X_test.shape))
print('X_overfit.shape: ' + str(X_overfit.shape))


weights_filenames = [f for f in os.listdir(weightsDir) if len(re.findall('.*weights.*\.hdf5$', f))]
current_iter = 0
for weights_filename in weights_filenames:
    fullpath = weightsDir + '/' + weights_filename
    print('Loading style_transfer with weights file: ' + fullpath)
    st_model = style_transfer_conv_transpose(fullpath)
    predict = K.function([st_model.input, K.learning_phase()], st_model.output)

    print('Predicting')
    # results_false = st_model.predict(X_test) # Equivalent to predict([X_test, False])
    results = predict([X_test, True])
    results_overfit = predict([X_overfit, True])

    print('Dumping results')
    for idx, im in enumerate(results):
        prefix = str(current_iter).zfill(4)
        fullOutPath = outputDir + '/' + prefix + "_" + str(idx) + ".png"
        print(im[0, 10:20, 10:20])
        save_image_st(fullOutPath, im)

        # fullFalsePath = outputDir + '/' + prefix + "_false.png"
        # save_image_st(fullFalsePath, results_false[idx])

        current_iter += 1

    for idx, im in enumerate(results_overfit):
        prefix = str(current_iter).zfill(4)
        fullOutPath = outputDir + '/' + prefix + "_overfit_" + str(idx) + ".png"
        save_image_st(fullOutPath, im)

        current_iter += 1
