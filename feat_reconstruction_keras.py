
# This file was a test to used more keras built-in function when training
# an input image, doesn't seem possible right now
# There seems to be an unsolvable problem in the function get_updates of
# keras optimizers

import os
import numpy as np

from keras import backend as K
from keras.engine import Input
from keras.optimizers import Adam

from vgg16.model import VGG_16_mean 
from vgg16.model_headless import *

from utils.imutils import *
from utils.lossutils import *
# from utils.optimizers import adam

dir = os.path.dirname(os.path.realpath(__file__))
vgg16Dir = dir + '/vgg16'
resultsDir = dir + '/models/results/vgg16'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)
dataDir = dir + '/data'

print('Loading a cat image')
X_train = load_image(dataDir + '/overfit/000.jpg')
print("X_train shape:", X_train.shape)

print('Loading Van Gogh')
vanGoghPath = dataDir + '/paintings/vangogh.jpg'
X_train_paint = np.array([load_image(vanGoghPath)])
print("X_train_paint shape:", X_train_paint.shape)

print('Loading mean')
meanPath = vgg16Dir + '/vgg-16_mean.npy'
mean = VGG_16_mean(path=meanPath)

print('Loading VGG headless 5')
modelWeights = vgg16Dir + '/vgg-16_headless_5_weights.hdf5'
vgg = VGG_16_headless_5(modelWeights, trainable=False, poolingType='average')
layer_dict = dict([(layer.name, layer) for layer in vgg.layers])
layers_names = [l for l in layer_dict if len(re.findall('conv_', l))]
layers_names.sort()

print('Building model')
input = Input((3, 256, 256))
outputs = vgg(input)

print('Building white noise images')
input_style_data = create_noise_tensor(3, 256, 256)
input_feat_data = create_noise_tensor(3, 256, 256)

current_iter = 1
for out in outputs:
    predict = K.function([input], [out])
    out_plabels = predict([X_train_paint - mean])
    out_ilabels = predict([X_train - mean])
    
    reg_TV = total_variation_error(input)

    for gamma in [1e-04, 1e-06, 0.]:
        print('Compiling VGG headless 1 for style reconstruction')
        loss_style = grams_frobenius_error(out_plabels[0], out)
        total_loss_style = loss_style + gamma * reg_TV
        # grads_style = K.gradients(total_loss_style, input_layer)
        # grads_style /= (K.sqrt(K.mean(K.square(grads_style))) + K.epsilon())
        adam = Adam(lr=1e-00)
        adam_updates = adam.get_updates([input], [], total_loss_style)
        iterate_style = K.function([input_style_data], [loss_style, grads_style], updates=adam_updates)

        # print('Compiling VGG headless 1 for ' + layer_name + ' feature reconstruction')
        # loss_feat = squared_normalized_euclidian_error(out_ilabels[0], out)
        # grads_feat = K.gradients(loss_feat + gamma * reg_TV, input_layer)[0]
        # grads_feat /= (K.sqrt(K.mean(K.square(grads_feat))) + K.epsilon())
        # iterate_feat = K.function([input_layer], [loss_feat, grads_feat])

        # prefix = 'biboup' + str(current_iter).zfill(4)
        # suffix = '_gamma' + str(gamma)

        # print('Training the image for style')
        # # config = {'learning_rate': 1e-00}
        # best_input_style_data = train_input(input_style_data - mean, iterate_style, adam, 600)
        # best_input_style_data += mean
        # fullOutPath = resultsDir + '/' + prefix + '_style_' + layer_name + suffix + ".png"
        # deprocess_image(fullOutPath, best_input_style_data[0])

        # print('Training the image for feature')
        # config = {'learning_rate': 1e-00}
        # best_input_feat_data = train_input(input_feat_data - mean, iterate_feat, adam, config, 600)
        # best_input_feat_data += mean
        # fullOutPath = resultsDir + '/' + prefix + '_feat_' + layer_name + suffix + ".png"
        # deprocess_image(fullOutPath, best_input_feat_data[0])

        # current_iter += 1

