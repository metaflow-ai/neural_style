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

channels = 3
width = 256
height = 256
input_shape = (channels, width, height)
batch = 4

print('Loading a cat image')
X_train = np.array([load_image(dataDir + '/overfit/000.jpg', size=(height, width))])
print("X_train shape:", X_train.shape)

print('Loading Van Gogh')
X_train_paint = load_images(dataDir + '/paintings', size=(height, width))[0:1, :, :, :]
print("X_train_paint shape:", X_train_paint.shape)

print('Loading mean')
meanPath = vgg16Dir + '/vgg-16_mean.npy'
mean = VGG_16_mean(path=meanPath)
print("mean shape:", X_train.shape)

print('Loading VGG headless 5')
modelWeights = vgg16Dir + '/vgg-16_headless_5_weights.hdf5'
model = VGG_16_headless_5(modelWeights, trainable=False)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layers_names = [l for l in layer_dict if len(re.findall('conv_', l))]
layers_names.sort()

input_layer = layer_dict['input'].input

print('Building white noise images')
input_style_data = create_noise_tensor(3, 256, 256)
input_feat_data = create_noise_tensor(3, 256, 256)

current_iter = 1
for layer_name in layers_names:
    print('Creating labels for ' + layer_name)
    out = layer_dict[layer_name].output
    predict = K.function([input_layer], out)

    # If we substract the mean in the training data, we need to substract the mean
    # in the input too. Also we seem to have a better result substracting the mean
    out_plabels = predict([X_train_paint - mean])
    # The mean is not needed to reproduce the original image
    out_ilabels = predict([X_train])
    
    reg_TV = total_variation_error(input_layer)

    for gamma in [1e-04, 1e-05, 1e-06]:
        print('Compiling VGG headless 1 for ' + layer_name + ' style reconstruction')
        loss_style = frobenius_error(grams(out_plabels), grams(out))
        grads_style = K.gradients(loss_style + gamma * reg_TV, input_layer)[0]
        grads_style /= (K.sqrt(K.mean(K.square(grads_style))) + K.epsilon())
        iterate_style = K.function([input_layer], [loss_style, grads_style])

        print('Compiling VGG headless 1 for ' + layer_name + ' feature reconstruction')
        loss_feat = squared_normalized_euclidian_error(out_ilabels, out)
        grads_feat = K.gradients(loss_feat + gamma * reg_TV, input_layer)[0]
        grads_feat /= (K.sqrt(K.mean(K.square(grads_feat))) + K.epsilon())
        iterate_feat = K.function([input_layer], [loss_feat, grads_feat])

        prefix = str(current_iter).zfill(4)
        suffix = '_gamma' + str(gamma)

        print('Training the image for style')
        config = {'learning_rate': 1e-00}
        best_input_style_data, style_losses = train_input(input_style_data - mean, iterate_style, adam, config, max_iter=600)
        fullOutPath = resultsDir + '/' + prefix + '_style_' + layer_name + suffix + ".png"
        deprocess_image(fullOutPath, best_input_style_data[0])
        plot_losses(style_losses, resultsDir, prefix + '_style', suffix)

        print('Training the image for feature')
        config = {'learning_rate': 1e-00}
        best_input_feat_data, feat_losses = train_input(input_feat_data, iterate_feat, adam, config, max_iter=600)
        fullOutPath = resultsDir + '/' + prefix + '_feat_' + layer_name + suffix + ".png"
        deprocess_image(fullOutPath, best_input_feat_data[0])
        plot_losses(feat_losses, resultsDir, prefix + '_feat', suffix)

        current_iter += 1
