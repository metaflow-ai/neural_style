import os, argparse, json, time, math
import numpy as np

from keras import backend as K

from vgg19.model_headless import VGG_19_headless_5, get_layer_data

from utils.imutils import (load_image, load_images, create_noise_tensor, 
                    deprocess, save_image, plot_losses, get_image_list)
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
parser.add_argument('--content', default=dataDir + '/train', type=str, help='Content image.')
parser.add_argument('--style', default=dataDir + '/paintings/edvard_munch-the_scream.jpg', type=str, help='Style image.')
parser.add_argument('--pooling_type', default='avg', type=str, choices=['max', 'avg'], help='VGG pooling type.')
parser.add_argument('--image_size', default=256, type=int, help='Input image size.')
parser.add_argument('--max_iter', default=500, type=int, help='Number of training iter.')
parser.add_argument('--input_type', default='content', type=str, choices=['random', 'content'], help='How to initialize the input data')
parser.add_argument('--print_inter_img', default=False, type=bool, help='Print intermediate images')
parser.add_argument('--output_dir', default=dataDir + '/output/vgg19/gatys_%s' % int(time.time()), type=str, help='optional output dir')
parser.add_argument('--no_dump_losses', default=False, type=bool, help='Dump a graph of the losses')
args = parser.parse_args()

output_dir = args.output_dir
if not os.path.isdir(output_dir): 
    os.makedirs(output_dir)

optimizer = 'adam'
if optimizer == 'lbfgs':
    K.set_floatx('float64') # scipy needs float64 to use lbfgs

channels = 3
width = args.image_size
height = args.image_size
input_shape = (channels, width, height)

if os.path.isdir(args.content):
    print('dir: %s' % args.content)
    image_list = get_image_list(args.content)
else:
    image_list = [args.content]
X_train = load_images(image_list, size=(height, width), dim_ordering='th', verbose=True)
print("X_train shape:", X_train.shape)

X_train_style = np.array([load_image(args.style, size=height, dim_ordering='th', verbose=True)])
print("X_train_style shape:", X_train_style.shape)

print('Loading VGG headless 5')
modelWeights = vgg19Dir + '/vgg-19_headless_5_weights.hdf5'
model = VGG_19_headless_5(modelWeights, trainable=False, pooling_type=args.pooling_type)
layer_dict, layers_names = get_layer_data(model, 'conv_')
print('Layers found:' + ', '.join(layers_names))

input_layer = model.input

# This didn't lead to any improvement
# layer_weights = json.load(open(dataDir + '/output/vgg19/reconstruction/layer_weights.json', 'r'))

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
batch_size = 8
nb_iter = int(math.floor(X_train.shape[0] / batch_size) + 1)
y_contents = []
print(nb_iter)
for i in range(nb_iter):
    if i == nb_iter - 1:
        y_contents_tmp = predict_content([X_train[i*batch_size:]])
    else:        
        y_contents_tmp = predict_content([X_train[i*batch_size:i*batch_size + batch_size]])

    for y_content_tmp_idx, y_contents_tmp in enumerate(y_contents):
        y_contents[y_content_tmp_idx].concatenate(y_contents_tmp)
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

alphas = [2e2]
betas = [1e0]
gammas = [1e-4]
config = {
    'style': args.style,
    'pooling_type': args.pooling_type,
    'max_iter': args.max_iter,
    'input_type': args.input_type,
    'optimizer': optimizer,
    'style_layers': style_layers,
    'content_layers': content_layers,
    'alpha': alphas,
    'beta': betas,
    'gamma': gammas,    
    'learning_rate': 5e-1
}
with open(output_dir + '/config.json', 'w') as outfile:
    json.dump(config, outfile)  

print('Using optimizer: ' + optimizer)
for file_idx in range(len(X_train)):
    print('Initializing input %s data' % args.input_type)
    if args.input_type == 'random':
        input_data = create_noise_tensor(height, width, channels, 'th')
    elif args.input_type == 'content':
        input_data = X_train[file_idx:file_idx+1, :, :, :].copy()
    else:
        raise Exception('Input type choices are random|content')

    current_iter = 1
    for idx, content_output in enumerate(content_output_layers):
        lc_name = content_layers[idx]
        train_content_loss = frobenius_error(y_contents[idx][file_idx:file_idx+1, :, :, :], content_output)
        print('Compiling VGG headless 5 for ' + lc_name + ' content reconstruction')
        # Those hyper parameters are selected thx to pre_analysis scripts
        # Made for avg pooling + content init
        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
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
                    filename_array = image_list[file_idx].split('/')[-1].split('.')

                    suffix = "_content%s_alpha%f_beta%f_gamma%f" % (lc_name, alpha, beta, gamma)
                    out_filename = filename_array[0] + suffix + '.' + filename_array[1]
                    def lambda_dump_input(obj):
                        current_iter = obj['current_iter']
                        input_data = obj['input_data']
                        if current_iter % 25 == 0 and args.print_inter_img == True:
                            save_image(output_dir + '/' + filename_array[0] + suffix + + '_' + str(current_iter.zfill(5)) + '.' + filename_array[1], deprocess(input_data[0], dim_ordering='th'))

                    best_input_data, losses = train_input(
                        input_data, 
                        train_iteratee, 
                        optimizer, 
                        {'learning_rate': config['learning_rate']}, 
                        max_iter=args.max_iter,
                        callbacks=[lambda_dump_input]
                    )

                    print('Dumping data')
                    save_image(output_dir + '/' + out_filename, deprocess(best_input_data[0], dim_ordering='th'))
                    if args.no_dump_losses == False:
                        plot_losses(losses, output_dir, filename_array[0], suffix)

                    current_iter += 1
