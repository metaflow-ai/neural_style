import os
import numpy as np
import json

from keras import backend as K
from keras.engine.training import collect_trainable_weights
from keras.layers.core import Lambda
from keras.optimizers import Adam
# from keras.utils.visualize_util import plot

from vgg16.model import VGG_16_mean 
from vgg16.model_headless import *
from models.style_transfer import *

from utils.imutils import *
from utils.lossutils import *

dir = os.path.dirname(os.path.realpath(__file__))
vgg16Dir = dir + '/vgg16'
resultsDir = dir + '/models/results/st'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)
dataDir = dir + '/data'
trainDir = dataDir + '/train'

channels = 3
width = 256
height = 256
input_shape = (channels, width, height)
batch = 4

print('Loading train images')
painting_fullpath = dataDir + '/paintings/van_gogh-starry_night_over_the_rhone.jpg'
X_style = np.array([load_image(painting_fullpath, (height, width))])
# X_train = load_images(trainDir, size=(height, width))
X_train = load_images(dataDir + '/overfit', size=(height, width))
print("X_train shape: " + str(X_train.shape))
print("X_style shape: " + str(X_style.shape))

print('Loading cross validation images')
# X_cv = load_images(dataDir + '/val')
X_cv = load_images(dataDir + '/overfit/cv', size=(height, width))
print("X_cv shape: " + str(X_cv.shape))

print('Loading mean')
meanPath = vgg16Dir + '/vgg-16_mean.npy'
mean = VGG_16_mean(path=meanPath)
print("mean shape: " + str(mean.shape))

print('Loading VGG headless 5 model')
modelWeights = vgg16Dir + '/vgg-16_headless_5_weights.hdf5'
vgg_model = VGG_16_headless_5(modelWeights, input_shape=input_shape, trainable=False, poolingType='average')

print('Loading style_transfer model')
stWeightsFullpath = dir + '/models/st_vangogh_weights.hdf5'
st_model = style_transfer(input_shape=input_shape)
init_weights = st_model.get_weights()
if os.path.isfile(stWeightsFullpath): 
    print("Loading weights")
    st_model = st_model.load_weights(stWeightsFullpath)

print('Building full model')
l_output = Lambda(lambda x: x - mean, output_shape=lambda shape: shape)(st_model.output)
[c11, c12, 
c21, c22, 
c31, c32, c33, 
c41, c42, c43,
c51, c52, c53] = vgg_model(l_output)

outputs = [c11, c12, c21, c22, c31, c32, c33, c41, c42, c43, c51, c52, c53]
outputs_layer_style = [c12, c22, c33, c43]
outputs_layer_feat = [c33]

predict_style = K.function([st_model.output], outputs_layer_style)
predict_feat = K.function([st_model.output], outputs_layer_feat)

print('Creating training labels')
style_labels = predict_style([X_style])
train_feat_labels = predict_feat([X_train])
if len(X_cv):
    cv_feat_labels = predict_feat([X_cv])

print('preparing loss functions')
loss_style1_2 = frobenius_error(grams(style_labels[0]), grams(outputs_layer_style[0]))
loss_style2_2 = frobenius_error(grams(style_labels[1]), grams(outputs_layer_style[1]))
loss_style3_3 = frobenius_error(grams(style_labels[2]), grams(outputs_layer_style[2]))
loss_style4_3 = frobenius_error(grams(style_labels[3]), grams(outputs_layer_style[3]))
train_loss_feat = squared_normalized_euclidian_error(train_feat_labels[0], outputs_layer_feat[0])
if len(X_cv):
    cv_loss_feat = squared_normalized_euclidian_error(cv_feat_labels[0], outputs_layer_feat[0])

reg_TV = total_variation_error(l_output)

print('Compiling VGG headless 5')
current_iter = 0
for alpha in [1e-02, 1e-03, 1e-04]:
    for beta in [1.]:
        for gamma in [1e-03, 1e-04, 1e-05]:
            print("alpha, beta, gamma:", alpha, beta, gamma)

            st_model.set_weights(init_weights)
            print('Preparing train iteratee function')
            train_loss = alpha * 0.25 * (loss_style1_2 + loss_style2_2 + loss_style3_3 + 1e03 * loss_style4_3) \
                + beta * train_loss_feat \
                + gamma * reg_TV
            adam = Adam(lr=1e-03)
            updates = adam.get_updates(collect_trainable_weights(st_model), st_model.constraints, train_loss)
            train_iteratee = K.function([st_model.input, K.learning_phase()], [train_loss], updates=updates)

            if len(X_cv):
                print('Preparing cv iteratee function')
                cv_loss = alpha * 0.2 * (loss_style1_2 + loss_style2_2 + loss_style3_3 + loss_style4_3 + loss_style5_3) \
                    + beta * cv_loss_feat \
                    + gamma * reg_TV
                cross_val_iteratee = K.function([st_model.input, K.learning_phase()], [cv_loss])
            else:
                cross_val_iteratee = None

            best_trainable_weights, losses = train_weights(
                X_train, 
                st_model, 
                train_iteratee, 
                cv_input_data=X_cv, 
                cross_val_iteratee=cross_val_iteratee, 
                max_iter=1500
            )

            prefix = str(current_iter).zfill(4)
            suffix = '_alpha' + str(alpha) +'_beta' + str(beta) + '_gamma' + str(gamma)
            st_weights = resultsDir + '/' + prefix + 'st_vangogh_weights' + suffix + '.hdf5'
            fullpath_loss = resultsDir + '/' + prefix + 'st_vangogh_loss' + suffix + '.json'
            current_iter += 1

            print("Saving final data")
            st_model.set_weights(best_trainable_weights)
            st_model.save_weights(st_weights, overwrite=True)

            with open(fullpath_loss, 'w') as outfile:
                json.dump(losses, outfile)  

            plot_losses(losses, resultsDir, prefix, suffix)
