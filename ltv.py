import os, argparse
import numpy as np

from keras import backend as K

from vgg19.model_headless import VGG_19_headless_5, get_layer_data

from utils.imutils import (load_image, create_noise_tensor, 
                    deprocess, save_image, plot_losses)
from utils.lossutils import (frobenius_error, grams, 
                    norm_l2, train_input, total_variation_error)

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
resultsDir = dataDir + '/output/vgg19/ltv'
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

layer_name = layers_names[3]
print('Creating labels for ' + layer_name)
out = layer_dict[layer_name].output
predict = K.function([input_layer], [out])
y_style = predict([X_train_style])[0]

print('Building white noise images')
input_data = create_noise_tensor(height, width, channels)

print('Using optimizer: ' + optimizer)
current_iter = 1
for gamma in [1e-7, 3e-7, 6e-7, 
            1e-6, 3e-6, 6e-6, 
            1e-5, 3e-5, 6e-5,
            1e-4, 3e-4, 6e-4]:
    print('gamma: %.7f' % gamma)
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
    save_image(fullOutPath, deprocess(best_input_style_data[0]))
    plot_losses(style_losses, resultsDir, prefix)

    current_iter += 1
