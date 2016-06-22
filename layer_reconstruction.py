from __future__ import absolute_import

import os, argparse, json
import numpy as np

from keras import backend as K

from vgg19.model_headless import VGG_19_headless_5, get_layer_data

from utils.imutils import (load_image, create_noise_tensor, 
                    dump_as_hdf5, deprocess, save_image,
                    plot_losses)
from utils.lossutils import (frobenius_error, grams, norm_l2, train_input)

if K._BACKEND == "tensorflow":
    K.set_image_dim_ordering('tf')
else:
    K.set_image_dim_ordering('th')

optimizer = 'adam'
if optimizer == 'lbfgs':
    K.set_floatx('float64') # scipy needs float64 to use lbfgs

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
dataDir = dir + '/data'
resultsDir = dataDir + '/output/vgg19/reconstruction'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)

parser = argparse.ArgumentParser(
    description='Neural artistic style. Generates an image by combining '
                'the content of an image and the style of another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--content', default=dataDir + '/overfit/COCO_val2014_000000000074.jpg', type=str, help='Content image.')
parser.add_argument('--style', default=dataDir + '/paintings/edvard_munch-the_scream.jpg', type=str, help='Style image.')
parser.add_argument('--pooling_type', default='avg', type=str, choices=['max', 'avg'], help='VGG pooling type.')
parser.add_argument('--image_size', default=256, type=int, help='Input image size.')
parser.add_argument('--max_iter', default=800, type=int, help='Number of training iter.')
args = parser.parse_args()


dim_ordering = K.image_dim_ordering()
print('dim_ordering: %s' % dim_ordering)
channels = 3
width = args.image_size
height = args.image_size
size = (height, width)
if dim_ordering == 'th':
    input_shape = (channels, width, height)
else:
    input_shape = (width, height, channels)

X_train = np.array([load_image(args.content, size=(height, width), verbose=True)])
print("X_train shape:", X_train.shape)

X_train_style = np.array([load_image(args.style, size=(height, width), verbose=True)])
print("X_train_style shape:", X_train_style.shape)

print('Loading VGG headless 5')
modelWeights = "%s/%s-%s-%s%s" % (vgg19Dir,'vgg-19', dim_ordering, K._BACKEND, '_headless_5_weights.hdf5')
model = VGG_19_headless_5(input_shape, modelWeights, trainable=False, pooling_type=args.pooling_type)
layer_dict, layers_names = get_layer_data(model, 'conv_')
print('Layers found:' + ', '.join(layers_names))

input_layer = model.input

print('Building white noise images')
input_data = create_noise_tensor(height, width, channels)

mean_losses = {}
print('Using optimizer: ' + optimizer)
current_iter = 1
for layer_name in layers_names:
    prefix = str(current_iter).zfill(4)

    print('Creating labels for ' + layer_name)
    out = layer_dict[layer_name].output
    predict = K.function([input_layer], [out])

    y_style = predict([X_train_style])[0]
    y_content = predict([X_train])[0]

    print('Compiling VGG headless 1 for ' + layer_name + ' style reconstruction')
    loss_style = frobenius_error(grams(y_style), grams(out))
    grads_style = K.gradients(loss_style, input_layer)
    if optimizer == 'adam':
        grads_style = norm_l2(grads_style)
    iterate_style = K.function([input_layer], [loss_style, grads_style])

    print('Training the image for style')
    config = {'learning_rate': 5e-1}
    best_input_style_data, style_losses = train_input(
        input_data, 
        iterate_style, 
        optimizer, 
        config, 
        max_iter=args.max_iter,
    )
    fullOutPath = resultsDir + '/' + prefix + '_style_' + layer_name  + ".png"
    save_image(fullOutPath, deprocess(best_input_style_data[0]))
    plot_losses(style_losses, resultsDir, prefix + '_style_')

    print('Compiling VGG headless 1 for ' + layer_name + ' content reconstruction')
    loss_content = frobenius_error(y_content, out)
    grads_content = K.gradients(loss_content, input_layer)
    if optimizer == 'adam':
        grads_content = norm_l2(grads_content)
    iterate_content = K.function([input_layer], [loss_content, grads_content])

    print('Training the image for content')
    config = {'learning_rate': 5e-1}
    best_input_content_data, content_losses = train_input(input_data, iterate_content, optimizer, config, max_iter=args.max_iter)
    fullOutPath = resultsDir + '/' + prefix + '_content_' + layer_name  + ".png"
    save_image(fullOutPath, deprocess(best_input_content_data[0]))
    plot_losses(content_losses, resultsDir, prefix + '_content_')

    mean_losses[layer_name] = {
        'style': {
            'min': min(style_losses['loss']),
            'mean': np.mean(style_losses['loss']),
            'max': max(style_losses['loss'])
        },
        'content': {
            'min': min(content_losses['loss']),
            'mean': np.mean(content_losses['loss']),
            'max': max(content_losses['loss'])
        }
    }

    current_iter += 1

with open(resultsDir + '/' + 'layer_weights.json', 'w') as outfile:
    json.dump(mean_losses, outfile)  
