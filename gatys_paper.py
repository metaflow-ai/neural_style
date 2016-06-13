import os, argparse, json, time
import numpy as np

from keras import backend as K

from vgg19.model_headless import VGG_19_headless_5, get_layer_data

from utils.imutils import (load_image, create_noise_tensor, 
                    dump_as_hdf5, deprocess, save_image,
                    plot_losses)
from utils.lossutils import (frobenius_error, total_variation_error, 
                            grams, norm_l2, train_input
                            )

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
dataDir = dir + '/data'
paintingsDir = dataDir + '/paintings'

parser = argparse.ArgumentParser(
    description='Neural artistic style. Generates an image by combining '
                'the content of an image and the style of another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--content', default=dataDir + '/overfit/000.jpg', type=str, help='Content image.')
parser.add_argument('--style', default=dataDir + '/paintings/edvard_munch-the_scream.jpg', type=str, help='Style image.')
parser.add_argument('--pooling_type', default='avg', type=str, choices=['max', 'avg'], help='VGG pooling type.')
parser.add_argument('--image_size', default=256, type=int, help='Input image size.')
parser.add_argument('--max_iter', default=600, type=int, help='Number of training iter.')
parser.add_argument('--input_type', default='random', type=str, choices=['random', 'content'], help='How to initialize the input data')
parser.add_argument('--print_inter_img', default=False, type=bool, help='Print intermediate images')
args = parser.parse_args()

resultsDir = dataDir + '/output/vgg19/gatys_%s_%s_%s' % (args.pooling_type, args.input_type, int(time.time()))
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)

optimizer = 'adam'
if optimizer == 'lbfgs':
    K.set_floatx('float64') # scipy needs float64 to use lbfgs

channels = 3
width = args.image_size
height = args.image_size
input_shape = (channels, width, height)

X_train = np.array([load_image(args.content, size=(height, width), dim_ordering='th', verbose=True)])
print("X_train shape:", X_train.shape)

X_train_style = np.array([load_image(args.style, size=height, dim_ordering='th', verbose=True)])
print("X_train_style shape:", X_train_style.shape)

print('Loading VGG headless 5')
modelWeights = vgg19Dir + '/vgg-19_headless_5_weights.hdf5'
model = VGG_19_headless_5(modelWeights, trainable=False, pooling_type=args.pooling_type)
layer_dict, layers_names = get_layer_data(model, 'conv_')
print('Layers found:' + ', '.join(layers_names))

input_layer = model.input

layer_weights = json.load(open(dataDir + '/output/vgg19/reconstruction/layer_weights.json', 'r'))

# Layer chosen thanks to the pre_analysis script
# conv_5_* layers doesn't hold enough information to rebuild the structure of the content/style
style_layers = ['conv_1_2', 'conv_2_2', 'conv_3_4', 'conv_4_2']
content_layers = ['conv_3_2']
style_output_layers = [layer_dict[ls_name].output for ls_name in style_layers]
content_output_layers = [layer_dict[lc_name].output for lc_name in content_layers]

print('Creating training labels')
predict_style = K.function([input_layer], style_output_layers)
y_styles = predict_style([X_train_style])
predict_content = K.function([input_layer], content_output_layers)
y_contents = predict_content([X_train])

print('Preparing training loss functions')
train_loss_styles = []
for idx, y_style in enumerate(y_styles):
    train_loss_styles.append(
        frobenius_error(
            grams(y_style), 
            grams(style_output_layers[idx])
        )
    )

reg_TV = total_variation_error(input_layer, 2)

print('Initializing input %s data' % args.input_type)
if args.input_type == 'random':
    input_data = create_noise_tensor(height, width, channels, 'th')
elif args.input_type == 'content':
    input_data = X_train.copy()
else:
    raise Exception('Input type choices are random|content')

print('Using optimizer: ' + optimizer)
current_iter = 1
for idx, content_output in enumerate(content_output_layers):
    lc_name = content_layers[idx]
    train_content_loss = frobenius_error(y_contents[idx], content_output)
    print('Compiling VGG headless 5 for ' + lc_name + ' content reconstruction')
    # Those hyper parameters are selected thx to pre_analysis scripts
    # Made for avg pooling + content init
    for alpha in [2e2]:
        for beta in [1e0]:
            for gamma in [1e-4]:
                print("alpha, beta, gamma:", alpha, beta, gamma)

                print('Computing train loss')
                tls = [alpha * train_loss_style / len(train_loss_styles) for style_idx, train_loss_style in enumerate(train_loss_styles)]
                tlc = beta * train_content_loss
                rtv = gamma * reg_TV
                train_loss =  sum(tls) + tlc + rtv

                print('Computing gradients')
                grads = K.gradients(train_loss, input_layer)
                if optimizer == 'adam':
                    grads = norm_l2(grads)
                inputs = [input_layer]
                outputs = [train_loss, grads, tlc] + tls

                print('Computing iteratee function')
                train_iteratee = K.function(inputs, outputs)

                config = {'learning_rate': 5e-1} #Can't go faster than that
                prefix = str(current_iter).zfill(4)
                suffix = "_alpha%f_beta%f_gamma%f" % (alpha, beta, gamma)
                filename = prefix + '_content' + lc_name + suffix
                def lambda_dump_input(obj):
                    current_iter = obj['current_iter']
                    input_data = obj['input_data']
                    if current_iter % 25 == 0 and args.print_inter_img == True:
                        save_image(resultsDir + '/' + filename + '_' + str(current_iter.zfill(5)) + '.png', deprocess(input_data[0], dim_ordering='th'))

                best_input_data, losses = train_input(
                    input_data, 
                    train_iteratee, 
                    optimizer, 
                    config, 
                    max_iter=args.max_iter,
                    callbacks=[lambda_dump_input]
                )

                print('Dumping data')
                # dump_as_hdf5(resultsDir + '/' + filename + ".hdf5", best_input_data[0])
                save_image(resultsDir + '/' + filename + '.png', deprocess(best_input_data[0], dim_ordering='th'))
                plot_losses(losses, resultsDir, prefix, suffix)

                current_iter += 1
