import os, json, gc, time, argparse

from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.optimizers import Adam

from keras.utils.visualize_util import plot as plot_model

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
results_dir = dir + '/models/data/st'
if not os.path.isdir(results_dir): 
    os.makedirs(results_dir)
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
parser.add_argument('--batch_size', default=4, type=int, help='batch size.')
parser.add_argument('--image_size', default=600, type=int, help='Input image size.')
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
vgg_model = VGG_19_headless_5(modelWeights, trainable=False, pooling_type=args.pooling_type, input_shape=input_shape)
layer_dict, layers_names = get_layer_data(vgg_model, 'conv_')

print('Selecting layers')
style_layers = ['conv_1_2', 'conv_2_2', 'conv_3_4', 'conv_4_2']
style_layers_mask = [name in style_layers for name in layers_names]
content_layers = ['conv_3_2']
content_layers_mask = [name in content_layers for name in layers_names]

print('Building full model')
mean = load_mean(name='vgg19', dim_ordering='th') # th ordering, BGR
preprocessed_output = Lambda(lambda x: x - mean, name="ltv")(st_model.output) # th, BGR ordering, sub mean
preds = vgg_model(preprocessed_output)
style_preds = mask_data(preds, style_layers_mask)
style_preds = [Lambda(lambda x: grams(x), lambda shape: (shape[0], shape[1], shape[1]))(style_pred) for style_pred in style_preds]
content_preds = mask_data(preds, content_layers_mask)
full_model = Model(input=[st_model.input], output=content_preds + style_preds + [preprocessed_output])

plot_model(full_model, to_file=results_dir + '/full_model.png', show_shapes=True)
plot_model(vgg_model, to_file=results_dir + '/vgg_model.png', show_shapes=True)


samples_per_epoch = len(get_image_list(trainDir))
style_fullpath_pefix = paintingsDir + '/results/' + args.style.split('/')[-1].split('.')[0]
train_generator = generate_data_from_image_list(
    trainDir, (height, width), style_fullpath_pefix, 
    VGG_19_headless_5(modelWeights, trainable=False, pooling_type=args.pooling_type, input_shape=input_shape),
    dim_ordering='th', verbose=False, st=True
)

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
            history = full_model.fit_generator(train_generator, 
                samples_per_epoch=samples_per_epoch, 
                max_q_size=args.batch_size, 
                nb_epoch=args.nb_epoch, 
                verbose=1
            )
            losses = history.history

            print("Saving final data")
            prefixedDir = results_dir + '/' + str(int(time.time()))
            export_model(st_model, prefixedDir)
            with open(prefixedDir + '/losses.json', 'w') as outfile:
                json.dump(losses, outfile)  
            plot_losses(losses, prefixedDir)
