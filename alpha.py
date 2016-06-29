import os, argparse, json
import numpy as np

from keras import backend as K

from vgg19.model_headless import VGG_19_headless_5, get_layer_data

from utils.imutils import (load_image, create_noise_tensor, 
                        save_image, plot_losses)
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
resultsDir = dataDir + '/output/vgg19/alpha'
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
parser.add_argument('--max_iter', default=600, type=int, help='Number of training iter.')
args = parser.parse_args()

dim_ordering = K.image_dim_ordering()
channels = 3
width = args.image_size
height = args.image_size
size = (height, width)
if dim_ordering == 'th':
    input_shape = (channels, width, height)
else:
    input_shape = (width, height, channels)

X_train = np.array([load_image(args.content, size=(height, width), preprocess_type='vgg19', verbose=True)])
print("X_train shape:", X_train.shape)

X_train_style = np.array([load_image(args.style, size=(height, width), preprocess_type='vgg19', verbose=True)])
print("X_train_style shape:", X_train_style.shape)

print('Loading VGG headless 5')
modelWeights = "%s/%s-%s-%s%s" % (vgg19Dir,'vgg-19', dim_ordering, K._BACKEND, '_headless_5_weights.hdf5')
model = VGG_19_headless_5(input_shape, modelWeights, trainable=False, pooling_type=args.pooling_type)
layer_dict, layers_names = get_layer_data(model, 'conv_')
print('Layers found:' + ', '.join(layers_names))

input_layer = model.input

layer_weights = json.load(open(dataDir + '/output/vgg19/reconstruction/layer_weights.json', 'r'))

print('Building white noise images')
input_data = create_noise_tensor(height, width, channels)

print('Using optimizer: ' + optimizer)
current_iter = 1
ls_name = lc_name = layers_names[3]
lc_name = layers_names[3]

for alpha in [1e0, 3e0, 6e0, 1e1, 3e1, 6e1, 1e2, 3e2, 6e2, 1e3, 3e3, 6e3]:
    print('Creating labels for content ' + lc_name + ' and style ' + ls_name)
    out_style = layer_dict[ls_name].output
    predict_style = K.function([input_layer], [out_style])
    y_style = predict_style([X_train_style])[0]

    out_content = layer_dict[lc_name].output
    predict_content = K.function([input_layer], [out_content])
    y_content = predict_content([X_train])[0]

    loss_style = frobenius_error(grams(y_style), grams(out_style))
    loss_content = frobenius_error(y_content, out_content)

    lc_weight = layer_weights[lc_name]['content']['mean']
    ls_weight = layer_weights[ls_name]['style']['mean']
    print("lc_weight: %f, ls_weight: %f" % (lc_weight, ls_weight))

    print('Compiling VGG headless 5 for content ' + lc_name + ' and style ' + ls_name)
    ls = alpha * loss_style / lc_weight
    lc = loss_content / ls_weight
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
    suffix = "_lc_weight%f_ls_weight%f" % (lc_weight, ls_weight)
    fullOutPath = resultsDir + '/' + prefix + '_style' + ls_name + '_content' + lc_name + suffix + ".png"
    # dump_as_hdf5(resultsDir + '/' + prefix + '_style' + ls_name + '_content' + lc_name + suffix + ".hdf5", best_input_data[0])
    save_image(fullOutPath, best_input_data[0], deprocess_type='vgg19')
    plot_losses(losses, resultsDir, prefix, suffix)

    current_iter += 1
