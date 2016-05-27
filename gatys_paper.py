import os

from keras import backend as K

from vgg19.model_headless import VGG_19_headless_5, get_layer_data

from utils.imutils import *
from utils.lossutils import *

optimizer = 'lbfgs'
if optimizer == 'lbfgs':
    K.set_floatx('float64') # scipy needs float64 to use lbfgs


dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
dataDir = dir + '/data'
resultsDir = dataDir + '/output/vgg19'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)

paintingsDir = dataDir + '/paintings'

channels = 3
width = 600
height = 600
input_shape = (channels, width, height)
batch = 4

print('Loading train images')
X_train = load_images(dataDir + '/overfit', size=(height, width), limit=1, dim_ordering='th')
print("X_train shape:", X_train.shape)

print('Loading VGG headless 5')
modelWeights = vgg19Dir + '/vgg-19_headless_5_weights.hdf5'
model = VGG_19_headless_5(modelWeights, trainable=False)
layer_dict, layers_names = get_layer_data(model, 'conv_')
print('Layers found:' + ', '.join(layers_names))

input_layer = model.input
style_layers_used = ['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_4_1', 'conv_5_1']
feat_layers_used = ['conv_1_2', 'conv_2_2', 'conv_3_2', 'conv_4_2', 'conv_5_2']
style_outputs_layer = [layer_dict[name].output for name in style_layers_used]
feat_outputs_layer = [layer_dict[name].output for name in feat_layers_used]

print('Creating training labels')
predict = K.function([input_layer], feat_outputs_layer)
train_feat_labels = predict([X_train])

print('Loading painting')
# suffix = "_ori.hdf5"
suffix = "_600x600.hdf5"
# suffix = "_256x256.hdf5"
painting_fullpath = paintingsDir + '/van_gogh-starry_night_over_the_rhone' + suffix 
y_styles = load_y_styles(painting_fullpath, style_layers_used)

print('Preparing training loss functions')
train_loss_style1 = frobenius_error(y_styles[0], grams(style_outputs_layer[0]))
train_loss_style2 = frobenius_error(y_styles[1], grams(style_outputs_layer[1]))
train_loss_style3 = frobenius_error(y_styles[2], grams(style_outputs_layer[2]))
train_loss_style4 = frobenius_error(y_styles[3], grams(style_outputs_layer[3]))
train_loss_style5 = frobenius_error(y_styles[4], grams(style_outputs_layer[4]))

reg_TV = total_variation_error(input_layer, 2)

print('Building white noise images')
input_data = create_noise_tensor(height, width, channels).transpose(0, 3, 1, 2).astype(K.floatx())

print('Using optimizer: ' + optimizer)
current_iter = 1
for idx, feat_output in enumerate(feat_outputs_layer):
    if idx != 3:
        # conv_1_2 and conv_2_2  are too "clean" for human perception
        # conv_3_4 tends to be too "clean"
        # conv_5_4 layer doesn't hold enough information to rebuild the structure of the photo
        continue
    layer_name_feat = feat_layers_used[idx]
    train_loss_feat = frobenius_error(train_feat_labels[idx], feat_output)
    print('Compiling VGG headless 5 for ' + layer_name_feat + ' feat reconstruction')
    for alpha in [1e2]:
        for beta in [5e0]:
            for gamma in [1e-3]:
                if alpha == beta and alpha != 1:
                    continue
                print("alpha, beta, gamma:", alpha, beta, gamma)

                print('Compiling model')
                tls1 = train_loss_style1 * alpha * 0.2
                tls2 = train_loss_style2 * alpha * 0.2
                tls3 = train_loss_style3 * alpha * 0.2
                tls4 = train_loss_style4 * alpha * 0.2
                tls5 = train_loss_style5 * alpha * 0.2
                tlf = train_loss_feat * beta
                rtv = reg_TV * gamma
                train_loss =  tls1 + tls2 + tls3 + tls4 + tls5 + tlf + rtv

                grads = K.gradients(train_loss, input_layer)
                if optimizer == 'adam':
                    grads = norm_l2(grads)
                train_iteratee = K.function([input_layer], [train_loss, grads, tls1, tls2, tls3, tls4, tls5, tlf])

                config = {'learning_rate': 5e-01}
                best_input_data, losses = train_input(input_data, train_iteratee, optimizer, config, max_iter=1000)

                prefix = str(current_iter).zfill(4)
                suffix = '_alpha' + str(alpha) +'_beta' + str(beta) + '_gamma' + str(gamma)
                filename = prefix + '_gatys_paper_feat' + layer_name_feat + suffix
                dump_as_hdf5(resultsDir + '/' + filename + ".hdf5", best_input_data[0])
                save_image(resultsDir + '/' + filename + '.png', deprocess(best_input_data[0], dim_ordering='th'))
                plot_losses(losses, resultsDir, prefix, suffix)

                current_iter += 1