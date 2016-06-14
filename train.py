import os, json, gc, time, argparse
import numpy as np

from keras import backend as K
from keras.engine.training import collect_trainable_weights
from keras.optimizers import Adam

from vgg19.model_headless import VGG_19_headless_5, get_layer_data
from models.style_transfer import (style_transfer_conv_transpose)

from utils.imutils import plot_losses, load_image, load_mean
from utils.lossutils import (grams, frobenius_error, 
                    train_weights, total_variation_error)
from utils.general import export_model, mask_data

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
parser.add_argument('--pooling_type', default='max', type=str, choices=['max', 'avg'], help='VGG pooling type.')
parser.add_argument('--batch_size', default=8, type=int, help='batch size.')
parser.add_argument('--image_size', default=256, type=int, help='Input image size.')
parser.add_argument('--max_iter', default=2500, type=int, help='Number of training iter.')
parser.add_argument('--nb_res_layer', default=6, type=int, help='Number of residual layers in the style transfer model.')
args = parser.parse_args()

channels = 3
width = args.image_size
height = args.image_size
input_shape = (channels, width, height)
batch_size = args.batch_size

X_train_style = np.array([load_image(args.style, size=height, dim_ordering='th', verbose=True)])
print("X_train_style shape:", X_train_style.shape)

print('Loading style_transfer model')
stWeightsFullpath = dir + '/models/st_weights.hdf5'
st_model = style_transfer_conv_transpose(input_shape=input_shape, nb_res_layer=args.nb_res_layer) # th ordering, BGR
if os.path.isfile(stWeightsFullpath): 
    print("Loading weights")
    st_model.load_weights(stWeightsFullpath)
init_weights = st_model.get_weights()
# plot_model(st_model, to_file=dir + '/st_model.png', show_shapes=True)

print('Loading VGG headless 5')
modelWeights = vgg19Dir + '/vgg-19_headless_5_weights.hdf5'
vgg_model = VGG_19_headless_5(modelWeights, trainable=False, pooling_type='max')
layer_dict, layers_names = get_layer_data(vgg_model, 'conv_')
print('Layers found:' + ', '.join(layers_names))

print('Selecting layers')
style_layers = ['conv_2_2', 'conv_3_2', 'conv_3_4', 'conv_4_3']
style_layers_mask = [name in style_layers for name in layers_names]
content_layers = ['conv_3_2']
content_layers_mask = [name in content_layers for name in layers_names]

print('Creating training labels')
style_outputs_layer = [grams(layer_dict[name].output) for name in style_layers]
predict_style = K.function([vgg_model.input], style_outputs_layer)
y_styles = predict_style([X_train_style]) # sub mean, th ordering, BGR

mean = load_mean(name='vgg19', dim_ordering='th') # th ordering, BGR
vgg_content_input = st_model.input - mean # th, BGR ordering, sub mean
vgg_content_output = vgg_model(vgg_content_input)
y_content = mask_data(vgg_model(vgg_content_input), content_layers_mask)
print(y_content)

print('Building full model')
preprocessed_output = st_model.output - mean # th, BGR ordering, sub mean
preds = vgg_model(preprocessed_output)
style_preds = mask_data(preds, style_layers_mask)
content_preds = mask_data(preds, content_layers_mask)

print('Preparing training loss functions')
train_loss_styles = []
for idx, y_style in enumerate(y_styles):
    train_loss_styles.append(
        frobenius_error(
            y_style, 
            grams(style_preds[idx])
        )
    )

train_loss_contents = []
for idx, y_content in enumerate(y_content):
    train_loss_contents.append(
        frobenius_error(y_content, content_preds[idx])
    )

reg_TV = total_variation_error(preprocessed_output, 2)

layer_weights = json.load(open(dataDir + '/output/vgg19/reconstruction/layer_weights.json', 'r'))

print('Iterating over hyper parameters')
current_iter = 0
for alpha in [1e1]:
    for beta in [1.]:
        for gamma in [5e-07]:
            print("alpha, beta, gamma:", alpha, beta, gamma)

            gc.collect()
        
            st_model.set_weights(init_weights)
            print('Compiling train loss')
            tls = [alpha * train_loss_style / layer_weights[style_layers[style_idx]]['style']['min'] for style_idx, train_loss_style in enumerate(train_loss_styles)]
            tlc = [beta * len(tls) * train_loss_content / layer_weights[content_layers[content_idx]]['content']['min'] for content_idx, train_loss_content in enumerate(train_loss_contents)]
            rtv = gamma * reg_TV
            train_loss =  sum(tls) + sum(tlc) + rtv

            print('Compiling Adam update')
            adam = Adam(lr=1e-03)
            updates = st_model.updates + adam.get_updates(collect_trainable_weights(st_model), st_model.constraints, train_loss)

            print('Compiling train function')
            inputs = [st_model.input, K.learning_phase()]
            outputs = [train_loss] + tlc + tls + [rtv] # Array concatenation
            train_iteratee = K.function(inputs, outputs, updates=updates)

            print('Starting training')
            weights, losses = train_weights(
                trainDir,
                # overfitDir, 
                (height, width),
                st_model, 
                train_iteratee, 
                max_iter=args.max_iter,
                batch_size=batch_size
            )

            best_weights = weights[0]
            last_weights = weights[1]

            print("Saving final data")
            prefixedDir = resultsDir + '/' + str(int(time.time()))
            export_model(st_model, prefixedDir, best_weights)
            with open(prefixedDir + '/losses.json', 'w') as outfile:
                json.dump(losses, outfile)  
            plot_losses(losses, prefixedDir)
