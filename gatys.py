import os
import numpy as np

from keras import backend as K

from vgg19.model import VGG_19_mean 
from vgg19.model_headless import *

from utils.imutils import *
from utils.lossutils import *
from utils.optimizers import adam

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
resultsDir = dir + '/models/results/vgg19'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)
dataDir = dir + '/data'

channels = 3
width = 256
height = 256
input_shape = (channels, width, height)
batch = 4

print('Loading a cat image')
X_train = load_images(dataDir + '/overfit', size=(height, width), limit=1, dim_ordering='th')
print("X_train shape:", X_train.shape)

print('Loading painting')
X_train_style = load_images(dataDir + '/paintings', size=(height, width), limit=1, dim_ordering='th')
print("X_train_style shape:", X_train_style.shape)

print('Loading VGG headless 5')
modelWeights = vgg19Dir + '/vgg-19_headless_5_weights.hdf5'
model = VGG_19_headless_5(modelWeights, trainable=False)
layer_dict, layers_names = get_layer_data(model, 'conv_')

input_layer = layer_dict['input'].input

print('Building white noise images')
input_data = create_noise_tensor(3, 256, 256)

current_iter = 1
for idx_feat, layer_name_feat in enumerate(layers_names):
    for idx_style, layer_name_style in enumerate(layers_names):
        print('Creating labels for feat ' + layer_name_feat + ' and style ' + layer_name_style)
        out_style = layer_dict[layer_name_style].output
        predict_style = K.function([input_layer], out_style)
        out_style_labels = predict_style([X_train_style])

        out_feat = layer_dict[layer_name_feat].output
        predict_feat = K.function([input_layer], out_feat)
        out_feat_labels = predict_feat([X_train])

        loss_style = frobenius_error(grams(out_style_labels), grams(out_style))
        loss_feat = squared_normalized_frobenius_error(out_feat_labels, out_feat)
        reg_TV = total_variation_error(input_layer)

        print('Compiling VGG headless 5 for feat ' + layer_name_feat + ' and style ' + layer_name_style)
        # At the same layer
        # if alpha/beta >= 1e02 we only see style
        # if alpha/beta <= 1e-04 we only see the picture
        for alpha in [1e02, 1., 1e-02, 1e-04]:
            for beta in [1.]:
                for gamma in [1e-03, 1e-04, 1e-05]:
                    if alpha == beta and alpha != 1:
                        continue
                    print("alpha, beta, gamma:", alpha, beta, gamma)

                    print('Compiling model')
                    loss = alpha * loss_style + beta * loss_feat + gamma * reg_TV
                    grads = K.gradients(loss, input_layer)[0]
                    grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())
                    iterate = K.function([input_layer], [loss, grads])

                    config = {'learning_rate': 1e-00}
                    best_input_data, losses = train_input(input_data, iterate, adam, config, max_iter=600)

                    prefix = str(current_iter).zfill(4)
                    suffix = '_alpha' + str(alpha) +'_beta' + str(beta) + '_gamma' + str(gamma)
                    fullOutPath = resultsDir + '/' + prefix + '_gatys_st' + layer_name_style + '_feat' + layer_name_feat + suffix + ".png"
                    deprocess(fullOutPath, best_input_data[0])
                    plot_losses(losses, resultsDir, prefix, suffix)

                    current_iter += 1
