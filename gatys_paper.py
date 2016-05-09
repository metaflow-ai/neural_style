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
model = VGG_16_headless_5(modelWeights, trainable=False)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
input_img = layer_dict['input'].input

layers_names = reversed([l for l in layer_dict if len(re.findall('conv_', l))])
max_iter = 1000

print('Building white noise images')
input_data = create_noise_tensor(3, 256, 256)

print('Creating labels')
out1_2 = layer_dict['conv_1_2'].output
out2_2 = layer_dict['conv_2_2'].output
out3_3 = layer_dict['conv_3_3'].output
out4_3 = layer_dict['conv_4_3'].output
out5_3 = layer_dict['conv_5_3'].output
predict = K.function([input_img], [out1_2, out2_2, out3_3, out4_3, out5_3])

out_plabels = predict([X_train_paint - mean])
out_ilabels = predict([X_train - mean])

print('Compiling VGG headless 1 for style + feat reconstruction')
for alpha in [0.2, 0.02, 0.002]:
    print("alpha:", alpha)
    print('Compiling model')
    loss_style1_2 = grams_frobenius_error(out_plabels[0], out1_2)
    loss_style2_2 = grams_frobenius_error(out_plabels[1], out2_2)
    loss_style3_3 = grams_frobenius_error(out_plabels[2], out3_3)
    loss_style4_3 = grams_frobenius_error(out_plabels[3], out4_3)
    loss_style5_3 = grams_frobenius_error(out_plabels[4], out5_3)
    loss_feat = euclidian_error(out_ilabels[1], out2_2)
    loss = alpha * (loss_style1_2 + loss_style2_2 + loss_style3_3 + loss_style4_3 + loss_style5_3) + loss_feat

    grads_style = K.gradients(loss, input_img)[0]
    grads_style /= (K.sqrt(K.mean(K.square(grads_style))) + 1e-5)
    iterate = K.function([input_img], [loss, grads_style])

    config = {'learning_rate': 1e-00}
    best_input_data = train_on_input(input_data - mean, iterate, adam, config)

    fullOutPath = resultsDir + '/gatys_paper_a' + str(alpha) + '.png'
    deprocess_image(best_input_data, fullOutPath)
