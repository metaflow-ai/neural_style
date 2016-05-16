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

print('Loading train images')
X_train_style = np.array([load_image(dataDir + '/paintings/vangogh.jpg')])
X_train = load_image(dataDir + '/overfit/000.jpg')
print("X_train shape:", X_train.shape)
print("X_train_style shape:", X_train_style.shape)

print('Loading cross validation images')
X_cv_style = np.array([load_image(dataDir + '/paintings/vangogh.jpg')])
X_cv = load_image(dataDir + '/overfit/001.jpg')


print('Loading mean')
meanPath = vgg16Dir + '/vgg-16_mean.npy'
mean = VGG_16_mean(path=meanPath)

print('Loading VGG headless 5')
modelWeights = vgg16Dir + '/vgg-16_headless_5_weights.hdf5'
model = VGG_16_headless_5(modelWeights, trainable=False, poolingType='average')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
input_layer = model.input
layers_used = ['conv_1_2', 'conv_2_2', 'conv_3_3', 'conv_4_3', 'conv_5_3']
outputs_layer = [layer_dict[name].output for name in layers_used]

print('Creating training labels')
predict = K.function([input_layer], outputs_layer)

train_style_labels = predict([X_train_style - mean])
train_feat_labels = predict([X_train - mean])

if len(X_cv_style):
    print('Creating cross validation labels')
    cv_style_labels = predict([X_cv_style - mean])
    cv_feat_labels = predict([X_cv - mean])

print('Preparing training loss functions')
train_loss_style1_2 = grams_frobenius_error(train_style_labels[0], outputs_layer[0])
train_loss_style2_2 = grams_frobenius_error(train_style_labels[1], outputs_layer[1])
train_loss_style3_3 = grams_frobenius_error(train_style_labels[2], outputs_layer[2])
train_loss_style4_3 = grams_frobenius_error(train_style_labels[3], outputs_layer[3])
train_loss_style5_3 = grams_frobenius_error(train_style_labels[4], outputs_layer[4])

train_losses_feat = []
# The first two are too "clean" for human perception
# train_losses_feat.append(squared_normalized_euclidian_error(train_feat_labels[0], outputs_layer[0]))
# train_losses_feat.append(squared_normalized_euclidian_error(train_feat_labels[1], outputs_layer[1]))

train_losses_feat.append(squared_normalized_euclidian_error(train_feat_labels[2], outputs_layer[2]))

# This one tend to be much more dreamy
train_losses_feat.append(squared_normalized_euclidian_error(train_feat_labels[3], outputs_layer[3]))

# The Fifth layer doesn't hold enough information to rebuild the structure of the photo
# train_losses_feat.append(squared_normalized_euclidian_error(train_feat_labels[4], outputs_layer[4]))

if len(X_cv_style):
    print('Preparing cross validation loss functions')
    cv_loss_style1_2 = grams_frobenius_error(cv_style_labels[0], outputs_layer[0])
    cv_loss_style2_2 = grams_frobenius_error(cv_style_labels[1], outputs_layer[1])
    cv_loss_style3_3 = grams_frobenius_error(cv_style_labels[2], outputs_layer[2])
    cv_loss_style4_3 = grams_frobenius_error(cv_style_labels[3], outputs_layer[3])
    cv_loss_style5_3 = grams_frobenius_error(cv_style_labels[4], outputs_layer[4])

    cv_losses_feat = []
    cv_losses_feat.append(squared_normalized_euclidian_error(cv_feat_labels[2], outputs_layer[2]))
    cv_losses_feat.append(squared_normalized_euclidian_error(cv_feat_labels[3], outputs_layer[3]))

reg_TV = total_variation_error(input_layer)

print('Building white noise images')
input_data = create_noise_tensor(3, 256, 256)
current_iter = 1

for idx, train_loss_feat in enumerate(train_losses_feat):
    layer_name_feat = layers_used[idx + 2]
    if len(X_cv_style):
        cv_loss_feat = cv_losses_feat[idx]
    print('Compiling VGG headless 5 for ' + layer_name_feat + ' feat reconstruction')
    for alpha in [1e-02, 1e-05]: # one for layer conv_3_3 and one for layer conv_4_3
        for beta in [1.]:
            for gamma in [1e-05, 1e-06, 1e-07]:
                if alpha == beta and alpha != 1:
                    continue
                print("alpha, beta, gamma:", alpha, beta, gamma)

                print('Compiling model')
                train_loss = alpha * 0.2 * (train_loss_style1_2 + train_loss_style2_2 + train_loss_style3_3 + train_loss_style4_3 + train_loss_style5_3) \
                    + beta * train_loss_feat \
                    + gamma * reg_TV

                grads = K.gradients(train_loss, input_layer)[0]
                grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())
                train_iteratee = K.function([input_layer], [train_loss, grads])

                if len(X_cv_style):
                    cv_loss = alpha * 0.2 * (cv_loss_style1_2 + cv_loss_style2_2 + cv_loss_style3_3 + cv_loss_style4_3 + cv_loss_style5_3) \
                        + beta * cv_loss_feat \
                        + gamma * reg_TV
                    cross_val_iteratee = K.function([input_layer], [cv_loss])
                else:
                    cross_val_iteratee = None

                config = {'learning_rate': 1e-01}
                best_input_data = train_input(input_data - mean, train_iteratee, adam, config, 4000, cross_val_iteratee)
                best_input_data += mean

                prefix = str(current_iter).zfill(4)
                suffix = '_alpha' + str(alpha) +'_beta' + str(beta) + '_gamma' + str(gamma)
                fullOutPath = resultsDir + '/' + prefix + '_gatys_paper_feat' + layer_name_feat + suffix + '.png'
                deprocess_image(fullOutPath, best_input_data[0])

                current_iter += 1