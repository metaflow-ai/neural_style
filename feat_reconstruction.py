import os
import numpy as np

from keras import backend as K

from vgg16.model import VGG_16_mean 
from vgg16.model_headless import *

from utils.imutils import *
from utils.lossutils import *

def adam(x, dx, config=None):
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', np.zeros_like(x))
  config.setdefault('v', np.zeros_like(x))
  config.setdefault('t', 0)
  
  next_x = None
  config['t'] += 1
  config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
  config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dx**2
  m_hat = config['m'] / (1 - config['beta1']**config['t'])
  v_hat = config['v'] / (1 - config['beta2']**config['t'])
  next_x = x - config['learning_rate'] * m_hat / (np.sqrt(v_hat) + config['epsilon'])

  return next_x, config

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

# http://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

layers_names = reversed([l for l in layer_dict if len(re.findall('conv_', l))])
max_iter = 1000
for layer_name in layers_names:
    print('Building white noise images')
    input_style_data = create_noise_tensor(3, 256, 256)
    input_feat_data = create_noise_tensor(3, 256, 256)

    print('Creating labels for ' + layer_name)
    out = layer_dict[layer_name].output
    predict = K.function([input_img], [out])

    out_plabels = predict([X_train_paint - mean])
    out_ilabels = predict([X_train - mean])

    print('Compiling VGG headless 1 for ' + layer_name + ' style reconstruction')
    loss_style = grams_frobenius_error(out_plabels[0], out)
    grads_style = K.gradients(loss_style, input_img)[0]
    grads_style /= (K.sqrt(K.mean(K.square(grads_style))) + 1e-5)
    iterate_style = K.function([input_img], [loss_style, grads_style])

    print('Compiling VGG headless 1 for ' + layer_name + ' feature reconstruction')
    loss_feat = euclidian_error(out_ilabels[0], out)
    grads_feat = K.gradients(loss_feat, input_img)[0]
    grads_feat /= (K.sqrt(K.mean(K.square(grads_feat))) + 1e-5)
    iterate_feat = K.function([input_img], [loss_feat, grads_feat])

    print('Training the image for style')
    input_style_data -= mean
    config = {'learning_rate': 1e-00}
    loss_style_val = 1000000000
    for i in range(max_iter):
        previous_loss_style_val = loss_style_val
        loss_style_val, grads_style_val = iterate_style([input_style_data])
        input_style_data, config = adam(input_style_data, grads_style_val, config)

        if i % 10 == 0:
            print(str(i) + ':', loss_style_val)

        if (np.abs(loss_style_val - previous_loss_style_val) < 0.1 and loss_style_val < 1):
            break
    print("final loss:", loss_style_val)
    fullOutPath = resultsDir + '/style_' + layer_name + ".png"
    deprocess_image(input_style_data[0], fullOutPath)

    print('Training the image for feature')
    input_feat_data -= mean
    config = {'learning_rate': 1e-00}
    loss_feat_val = 1000000000
    for i in range(max_iter):
        previous_loss_feat_val = loss_feat_val
        loss_feat_val, grads_feat_val = iterate_feat([input_feat_data])
        input_feat_data, config = adam(input_feat_data, grads_feat_val, config)

        if i % 10 == 0:
            print(str(i) + ':', loss_feat_val)

        # if i % 20 == 0:
        #     deprocess_image(input_feat_data[0], resultsDir + '/feat_' + layer_name + "_" + str(i) + ".png")

        if (np.abs(previous_loss_feat_val - loss_feat_val) < 0.1 and loss_feat_val < 0.1):
            break
    print("final loss:", loss_feat_val)
    fullOutPath = resultsDir + '/feat_' + layer_name + ".png"
    deprocess_image(input_feat_data[0], fullOutPath)

