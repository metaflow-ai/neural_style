import os, argparse, sys

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

import numpy as np
import h5py
from keras import backend as K

from vgg19.model_headless import VGG_19_headless_5, get_layer_data

from utils.imutils import (load_image, load_images, create_noise_tensor, 
                    deprocess, save_image, plot_losses, get_image_list)
from utils.lossutils import (frobenius_error, total_variation_error, 
                            grams, norm_l2, train_input
                            )

dataDir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dataDir + '/../vgg19'

parser = argparse.ArgumentParser(
    description='Neural artistic style. Generates an image by combining '
                'the content of an image and the style of another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--style_folder', default=dataDir + '/paintings', type=str, help='Style folder')
parser.add_argument('--content_folder', default=dataDir + '/train', type=str, help='Content folder')
parser.add_argument('--pooling_type', default='max', type=str, choices=['max', 'avg'], help='VGG pooling type.')
parser.add_argument('--image_size', default=600, type=int, help='Input image size.')
args = parser.parse_args()

if args.content_folder != '':
    results_content_dir = args.content_folder + '/results'
    if not os.path.isdir(results_content_dir): 
        os.makedirs(results_content_dir)
if args.style_folder != '':
    results_style_dir = args.style_folder + '/results'
    if not os.path.isdir(results_style_dir): 
        os.makedirs(results_style_dir)

channels = 3
width = args.image_size
height = args.image_size
input_shape = (channels, width, height)

print('Loading VGG model')
modelWeights = vgg19Dir + '/vgg-19_headless_5_weights.hdf5'
model = VGG_19_headless_5(modelWeights, trainable=False, pooling_type=args.pooling_type)
layer_dict, layers_names = get_layer_data(model, 'conv_')

print('Compiling predict functions')
style_layers = ['conv_1_2', 'conv_2_2', 'conv_3_4', 'conv_4_2']
style_output_layers = [grams(layer_dict[ls_name].output) for ls_name in style_layers]
predict_style = K.function([model.input], style_output_layers)

content_layers = ['conv_3_2']
content_output_layers = [layer_dict[lc_name].output for lc_name in content_layers]
predict_content = K.function([model.input], content_output_layers)

if 'results_style_dir' in locals():
    image_list = get_image_list(args.style_folder)
    for image_path in image_list:
        X_train_style = np.array([load_image(image_path, size=height, dim_ordering='th', verbose=True)])
        results = predict_style([X_train_style])

        filename = image_path.split('/')[-1].split('.')[0]
        output_filename = results_style_dir + '/' + filename + '_' + str(args.image_size) + '.hdf5'
        with h5py.File(output_filename, 'w') as hf:
            for idx, style_layer in enumerate(style_layers):
                hf.create_dataset(style_layer, data=results[idx])

if 'results_content_dir' in locals():
    image_list = get_image_list(args.content_folder)
    for image_path in image_list:
        X_train_content = np.array([load_image(image_path, size=(height, width), dim_ordering='th', verbose=True)])
        results = predict_content([X_train_content])

        filename = image_path.split('/')[-1].split('.')[0]
        output_filename = results_content_dir + '/' + filename + '_' + str(args.image_size) + '.hdf5'
        with h5py.File(output_filename, 'w') as hf:
            for idx, content_layer in enumerate(content_layers):
                hf.create_dataset(content_layer, data=results[idx])




