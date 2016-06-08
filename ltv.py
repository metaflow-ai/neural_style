import os, argparse
import numpy as np

from keras import backend as K

from vgg19.model_headless import VGG_19_headless_5, get_layer_data

from utils.imutils import (load_image, create_noise_tensor, 
                    dump_as_hdf5, deprocess, save_image,
                    plot_losses)
from utils.lossutils import (frobenius_error, grams, norm_l2, train_input, total_variation_error)

optimizer = 'adam'
if optimizer == 'lbfgs':
    K.set_floatx('float64') # scipy needs float64 to use lbfgs

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
dataDir = dir + '/data'
resultsDir = dataDir + '/output/vgg19/ltv'
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
layer_dict, layers_names = get_layer_data(model, 'conv_')
print('Layers found:' + ', '.join(layers_names))

input_layer = model.input

layer_name = layers_names[1]
print('Creating labels for ' + layer_name)
out = layer_dict[layer_name].output
predict = K.function([input_layer], [out])
y_style = predict([X_train_style])[0]

print('Building white noise images')
input_data = create_noise_tensor(height, width, channels, 'th')

print('Using optimizer: ' + optimizer)
current_iter = 1
for gamma in [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]:
    print('gamma: %f' % gamma)
    prefix = str(current_iter).zfill(4)

    print('Compiling gamma')
    loss_style = frobenius_error(grams(y_style), grams(out))
    reg_TV = total_variation_error(input_layer, 2)
    rtv = gamma * reg_TV
    total_loss = loss_style + rtv
    grads_style = K.gradients(total_loss, input_layer)
    if optimizer == 'adam':
        grads_style = norm_l2(grads_style)
    iterate_style = K.function([input_layer], [total_loss, grads_style, loss_style, rtv])

    print('Training the image for style')
    config = {'learning_rate': 5e-1}
    best_input_style_data, style_losses = train_input(
        input_data, 
        iterate_style, 
        optimizer, 
        config, 
        max_iter=args.max_iter,
    )
    fullOutPath = resultsDir + '/' + prefix + '_gamma' + str(gamma)  + ".png"
    save_image(fullOutPath, deprocess(best_input_style_data[0], dim_ordering='th'))
    plot_losses(style_losses, resultsDir, prefix)

    current_iter += 1
