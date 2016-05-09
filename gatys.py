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

print('Loading VGG headless 3')
modelWeights = vgg16Dir + '/vgg-16_headless_3_weights.hdf5'
model = VGG_16_headless_3(modelWeights, trainable=False)
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
        loss_feat = euclidian_error(out_feat_labels[0], out_feat)

        print('Compiling VGG headless 3 for ' + layer_name_style + ' style, ' + layer_name_feat + ' feat reconstruction')
        for alpha in [1e04, 1e02, 1., 1e-02, 1e-04]:
            print("alpha:", alpha)

            print('Compiling model')
            loss = alpha * loss_style + loss_feat
            grads = K.gradients(loss, input_layer)[0]
            # grads = K.gradients(loss_style, input_layer)[0]
            grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
            iterate = K.function([input_layer], [loss, grads])

            config = {'learning_rate': 1e-00}
            best_input_data = train_on_input(input_data - mean, iterate, adam, config)
            best_input_data += mean

            prefix = str(current_iter).zfill(4)
            fullOutPath = resultsDir + '/' + prefix + '_gatys_st' + layer_name_style + '_feat' + layer_name_feat + "_alpha" + str(alpha) + ".png"
            deprocess_image(best_input_data[0], fullOutPath)

            current_iter += 1
