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
paintingsDir = dataDir + '/paintings'

channels = 3
width = 256
height = 256
input_shape = (channels, width, height)
batch = 4

print('Loading train images')
X_train = np.array([load_image(dataDir + '/overfit/000.jpg', size=(height, width))])
print("X_train shape:", X_train.shape)

print('Loading mean')
meanPath = vgg16Dir + '/vgg-16_mean.npy'
mean = VGG_16_mean(path=meanPath)

print('Loading VGG headless 5')
modelWeights = vgg16Dir + '/vgg-16_headless_5_weights.hdf5'
model = VGG_16_headless_5(modelWeights, trainable=False, poolingType='average')
layer_dict = dict([(layer.name, layer) for layer in model.layers])
input_layer = model.input
layers_used = ['conv_1_2', 'conv_2_2', 'conv_3_3', 'conv_4_3']# , 'conv_5_3']
outputs_layer = [layer_dict[name].output for name in layers_used]

print('Creating training labels')
predict = K.function([input_layer], outputs_layer)
train_feat_labels = predict([X_train - mean])

print('Loading painting')
X_train_style = np.array([load_image(paintingsDir + '/van_gogh-starry_night_over_the_rhone.jpg', size=(height, width))])
train_style_labels = predict([X_train_style - mean])
y_styles = []
y_styles.append(grams(train_style_labels[0]))
y_styles.append(grams(train_style_labels[1]))
y_styles.append(grams(train_style_labels[2]))
y_styles.append(grams(train_style_labels[3]))

# suffix = "_ori.hdf5"
# # suffix = "_600x600.hdf5"
# # suffix = "_256x256.hdf5"
# painting_fullpath = paintingsDir + '/van_gogh-starry_night_over_the_rhone' + suffix 
# with h5py.File(painting_fullpath, 'r') as f:
#     y_styles = []
#     y_styles.append(f['conv_1_2'][()])
#     y_styles.append(f['conv_2_2'][()])
#     y_styles.append(f['conv_3_3'][()])
#     y_styles.append(f['conv_4_3'][()])
#     y_styles.append(f['conv_5_3'][()])
    

print('Preparing training loss functions')
train_loss_style1_2 = frobenius_error(y_styles[0], grams(outputs_layer[0]))
train_loss_style2_2 = frobenius_error(y_styles[1], grams(outputs_layer[1]))
train_loss_style3_3 = frobenius_error(y_styles[2], grams(outputs_layer[2]))
train_loss_style4_3 = frobenius_error(y_styles[3], grams(outputs_layer[3]))

# This input allow too much large shape of style inputs project back on the final output
# train_loss_style5_3 = frobenius_error(grams(train_style_labels[4]), grams(outputs_layer[4]))

train_losses_feat = []
# The first two are too "clean" for human perception
# train_losses_feat.append(squared_normalized_euclidian_error(train_feat_labels[0], outputs_layer[0]))

# This one tends to be too "clean"
train_losses_feat.append(squared_normalized_euclidian_error(train_feat_labels[1], outputs_layer[1]))

train_losses_feat.append(squared_normalized_euclidian_error(train_feat_labels[2], outputs_layer[2]))

# This one tends to be too much more dreamy
# train_losses_feat.append(squared_normalized_euclidian_error(train_feat_labels[3], outputs_layer[3]))

# The Fifth layer doesn't hold enough information to rebuild the structure of the photo
# train_losses_feat.append(squared_normalized_euclidian_error(train_feat_labels[4], outputs_layer[4]))

reg_TV = total_variation_error(input_layer)

print('Building white noise images')
input_data = create_noise_tensor(channels, height, width)
current_iter = 1

for idx, train_loss_feat in enumerate(train_losses_feat):
    layer_name_feat = layers_used[idx + 1]
    print('Compiling VGG headless 5 for ' + layer_name_feat + ' feat reconstruction')
    for alpha in [1e-02, 1e-03]:
        for beta in [1.]:
            for gamma in [1e-02, 1e-04, 1e-06]:
                if alpha == beta and alpha != 1:
                    continue
                print("alpha, beta, gamma:", alpha, beta, gamma)

                print('Compiling model')
                # Based on previous analysis, the conv_2_2/conv_3_3 layers have
                train_loss = alpha * 0.25 * (train_loss_style1_2 + train_loss_style2_2 + train_loss_style3_3 + 1e03 * train_loss_style4_3) \
                    + beta * train_loss_feat \
                    + gamma * reg_TV

                grads = K.gradients(train_loss, input_layer)[0]
                grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())
                train_iteratee = K.function([input_layer], [train_loss, grads])

                config = {'learning_rate': 1e-01}
                best_input_data, losses = train_input(input_data - mean, train_iteratee, adam, config, max_iter=2000)

                prefix = str(current_iter).zfill(4)
                suffix = '_alpha' + str(alpha) +'_beta' + str(beta) + '_gamma' + str(gamma)
                fullOutPath = resultsDir + '/' + prefix + '_gatys_paper_feat' + layer_name_feat + suffix + '.png'
                deprocess_image(fullOutPath, best_input_data[0])
                plot_losses(losses, resultsDir, prefix, suffix)

                current_iter += 1