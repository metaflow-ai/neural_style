import os, argparse, json
import numpy as np

from keras import backend as K

from vgg19.model_headless import VGG_19_headless_5, get_layer_data

from utils.imutils import (load_image, create_noise_tensor, 
                    dump_as_hdf5, deprocess, save_image,
                    plot_losses)
from utils.lossutils import (frobenius_error, grams, norm_l2, train_input)

optimizer = 'adam'
if optimizer == 'lbfgs':
    K.set_floatx('float64') # scipy needs float64 to use lbfgs

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
dataDir = dir + '/data'
resultsDir = dataDir + '/output/vgg19/influence'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)

parser = argparse.ArgumentParser(
    description='Neural artistic style. Generates an image by combining '
                'the content of an image and the style of another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--content', default=dataDir + '/overfit/000.jpg', type=str, help='Content image.')
parser.add_argument('--style', default=dataDir + '/paintings/edvard_munch-the_scream.jpg', type=str, help='Style image.')
parser.add_argument('--pooling_type', default='max', type=str, choices=['max', 'avg'], help='VGG pooling type.')
parser.add_argument('--image_size', default=256, type=int, help='Input image size.')
parser.add_argument('--max_iter', default=600, type=int, help='Number of training iter.')
args = parser.parse_args()

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
layer_dict, layers_names = get_layer_data(model, 'conv_(1|2|3|4)') # remove conv_5_* layers
print('Layers found:' + ', '.join(layers_names))

input_layer = model.input

layer_weights = json.load(open(dataDir + '/output/vgg19/reconstruction/layer_weights.json', 'r'))

print('Building white noise images')
input_data = create_noise_tensor(height, width, channels, 'th')

print('Using optimizer: ' + optimizer)
current_iter = 1
alpha = 5e1 # 1e2
for idx_content, lc_name in enumerate(layers_names):
    for idx_style, ls_name in enumerate(layers_names):
        print('Creating labels for content ' + lc_name + ' and style ' + ls_name)
        out_style = layer_dict[ls_name].output
        predict_style = K.function([input_layer], [out_style])
        y_style = predict_style([X_train_style])[0]

        out_content = layer_dict[lc_name].output
        predict_content = K.function([input_layer], [out_content])
        y_content = predict_content([X_train])[0]

        loss_style = frobenius_error(grams(y_style), grams(out_style))
        loss_content = frobenius_error(y_content, out_content)

        print("weight_lc: %f, weight_ls: %f" % (layer_weights[lc_name]['content'], layer_weights[ls_name]['style']))

        print('Compiling VGG headless 5 for content ' + lc_name + ' and style ' + ls_name)
        ls = alpha * loss_style / layer_weights[ls_name]['style']
        lc = loss_content / layer_weights[lc_name]['content']
        loss =  ls + lc
        grads = K.gradients(loss, input_layer)
        if optimizer == 'adam':
            grads = norm_l2(grads)
        iterate = K.function([input_layer], [loss, grads, lc, ls])

        print('Training the image')
        config = {'learning_rate': 5e-1}
        best_input_data, losses = train_input(
            input_data, 
            iterate, 
            optimizer, 
            config, 
            max_iter=args.max_iter
        )
        prefix = str(current_iter).zfill(4)
        suffix = "_weight_lc%f_weight_ls%f" % (layer_weights[lc_name]['content'], layer_weights[ls_name]['style'])
        fullOutPath = resultsDir + '/' + prefix + '_style' + ls_name + '_content' + lc_name + suffix + ".png"
        # dump_as_hdf5(resultsDir + '/' + prefix + '_style' + ls_name + '_content' + lc_name + suffix + ".hdf5", best_input_data[0])
        save_image(fullOutPath, deprocess(best_input_data[0], dim_ordering='th'))
        plot_losses(losses, resultsDir, prefix, suffix)

        current_iter += 1
