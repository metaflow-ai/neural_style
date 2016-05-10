import os
import numpy as np

from keras import backend as K

from vgg16.model import VGG_16_mean 
from vgg16.model_headless import *

from utils.imutils import *
from utils.lossutils import *
from utils.optimizers import adam

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
model = VGG_16_headless_5(modelWeights, trainable=False, poolingType='average')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layers_names = [l for l in layer_dict if len(re.findall('conv_', l))]
layers_names.sort()

input_layer = layer_dict['input'].input

print('Building white noise images')
input_data = create_noise_tensor(3, 256, 256)

current_iter = 1
for layer_name_feat in layers_names:
    for layer_name_style in layers_names:
        print('Creating labels for feat ' + layer_name_feat + ' and style ' + layer_name_style)
        out_style = layer_dict[layer_name_style].output
        predict_style = K.function([input_layer], [out_style])
        out_style_labels = predict_style([X_train_paint - mean])

        out_feat = layer_dict[layer_name_feat].output
        predict_feat = K.function([input_layer], [out_feat])
        out_feat_labels = predict_feat([X_train - mean])

        loss_style = grams_frobenius_error(out_style_labels[0], out_style)
        loss_feat = squared_nornalized_euclidian_error(out_feat_labels[0], out_feat)
        reg_TV = total_variation_error(input_data)

        print('Compiling VGG headless 5 for feat ' + layer_name_feat + ' and style ' + layer_name_style)
        # At the same layer
        # if alpha/beta >= 1e02 we only see style
        # if alpha/beta <= 1e-04 we only see the picture
        for alpha in [1., 1e-02, 1e-04]:
            for beta in [1.]:
                for gamma in [1, 1e-04, 0]:
                    if alpha == beta and alpha != 1:
                        continue
                    print("alpha, beta, gamma:", alpha, beta, gamma)

                    print('Compiling model')
                    loss = alpha * loss_style + beta * loss_feat + gamma * reg_TV
                    grads = K.gradients(loss, input_layer)[0]
                    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
                    iterate = K.function([input_layer], [loss, grads])

                    config = {'learning_rate': 1e-00}
                    best_input_data = train_on_input(input_data - mean, iterate, adam, config)
                    best_input_data += mean

                    prefix = str(current_iter).zfill(4)
                    suffix = '_alpha' + str(alpha) +'_beta' + str(beta) + '_gamma' + str(gamma)
                    fullOutPath = resultsDir + '/' + prefix + '_gatys_st' + layer_name_style + '_feat' + layer_name_feat + suffix + ".png"
                    deprocess_image(best_input_data[0], fullOutPath)

                    current_iter += 1
