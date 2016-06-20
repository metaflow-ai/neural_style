import os, json, gc, time, argparse
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.optimizers import Adam

from vgg19.model_headless import VGG_19_headless_5, get_layer_data
from models.style_transfer import (style_transfer_conv_transpose
                                , style_transfer_conv_inception_3)

from utils.imutils import plot_losses, load_mean, get_image_list
from utils.lossutils import (grams, frobenius_error, 
                    train_weights, total_variation_error_keras)
from utils.general import export_model, mask_data, generate_data_from_image_list

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
dataDir = dir + '/data'
resultsDir = dir + '/models/data/st'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)
trainDir = dataDir + '/train'
overfitDir = dataDir + '/overfit'
paintingsDir = dataDir + '/paintings'

parser = argparse.ArgumentParser(
    description='Neural artistic style. Generates an image by combining '
                'the content of an image and the style of another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--style', default=dataDir + '/paintings/edvard_munch-the_scream.jpg', type=str, help='Style image.')
parser.add_argument('--weights', default='', type=str, help='Load pretrained weights')
parser.add_argument('--pooling_type', default='avg', type=str, choices=['max', 'avg'], help='VGG pooling type.')
parser.add_argument('--batch_size', default=8, type=int, help='batch size.')
parser.add_argument('--image_size', default=256, type=int, help='Input image size.')
parser.add_argument('--nb_epoch', default=2, type=int, help='Number of epoch.')
parser.add_argument('--nb_res_layer', default=6, type=int, help='Number of residual layers in the style transfer model.')
args = parser.parse_args()

channels = 3
width = args.image_size
height = args.image_size
input_shape = (channels, width, height)
batch_size = args.batch_size

print('Loading style_transfer model')
st_model = style_transfer_conv_inception_3(mode=2, input_shape=input_shape, nb_res_layer=args.nb_res_layer) # th ordering, BGR
if os.path.isfile(args.weights): 
    print("Loading weights")
    st_model.load_weights(args.weights)
init_weights = st_model.get_weights()

print('Loading VGG headless 5')
modelWeights = vgg19Dir + '/vgg-19_headless_5_weights.hdf5'
vgg_model = VGG_19_headless_5(modelWeights, trainable=False, pooling_type=args.pooling_type)

print('Selecting layers')
style_layers_mask = ['conv_1_2', 'conv_2_2', 'conv_3_4', 'conv_4_2']
content_layers_mask = ['conv_3_2']

print('Building full model')
mean = load_mean(name='vgg19', dim_ordering='th') # th ordering, BGR
preprocessed_output = Lambda(lambda x: x - mean)(st_model.output) # th, BGR ordering, sub mean
preds = vgg_model(preprocessed_output)
style_preds = mask_data(preds, style_layers_mask)

def lambda_forward(x):
    return grams(x)
def lambda_get_output_shape(input_shape):
    return (input_shape[0], input_shape[1], input_shape[1])
grams_layer = Lambda(lambda_forward, lambda_get_output_shape)

style_preds = [grams_layer(style_pred) for style_pred in style_preds]
content_preds = mask_data(preds, content_layers_mask)
full_model = Model(input=[st_model.input], output=content_preds + style_preds + [preprocessed_output])

samples_per_epoch = len(get_image_list(trainDir))
style_fullpath_pefix = paintingsDir + '/results/' + args.style.split('/')[-1].split('.')[0]
generator = generate_data_from_image_list(trainDir, (height, width), style_fullpath_pefix, vgg_model)

print('Iterating over hyper parameters')
current_iter = 0
for alpha in [2e2]:
    for beta in [1.]:
        for gamma in [1e-4]:
            print("alpha, beta, gamma:", alpha, beta, gamma)

            gc.collect()
            
            st_model.set_weights(init_weights)

            print('Compiling model')
            adam = Adam(lr=1e-03)
            full_model.compile(optimizer=adam,
                loss=[
                    # content
                    'mse',
                    # style
                    'mse',
                    'mse',
                    'mse',
                    'mse',
                    # ltv
                    total_variation_error_keras
                ],
                loss_weights=[
                    # content
                    beta,
                    # style
                    alpha / len(style_preds),
                    alpha / len(style_preds),
                    alpha / len(style_preds),
                    alpha / len(style_preds),
                    # ltv
                    gamma
                ]
            )

            print('Training model')
            history = full_model.fit_generator(generator, 
                samples_per_epoch=samples_per_epoch, 
                max_q_size=args.batch_size, 
                nb_epoch=args.nb_epoch, 
                verbose=1
            )
            losses = history.history

            print("Saving final data")
            prefixedDir = resultsDir + '/' + str(int(time.time()))
            export_model(st_model, prefixedDir)
            with open(prefixedDir + '/losses.json', 'w') as outfile:
                json.dump(losses, outfile)  
            plot_losses(losses, prefixedDir)
