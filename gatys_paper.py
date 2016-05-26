import os

from keras import backend as K

from vgg19.model_headless import VGG_19_headless_5, get_layer_data

from utils.imutils import *
from utils.lossutils import *
from utils.optimizers import adam

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
resultsDir = dir + '/models/results/vgg19'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)
dataDir = dir + '/data'
paintingsDir = dataDir + '/paintings'

channels = 3
width = 512
height = 512
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
feat_layers_used = ['conv_1_1', 'conv_2_1', 'conv_3_1', 'conv_4_1', 'conv_5_1']
style_outputs_layer = [layer_dict[name].output for name in style_layers_used]
feat_outputs_layer = [layer_dict[name].output for name in feat_layers_used]

print('Creating training labels')
predict = K.function([input_layer], feat_outputs_layer)
train_feat_labels = predict([X_train])

print('Loading painting')
suffix = "_ori.hdf5"
# suffix = "_600x600.hdf5"
# suffix = "_256x256.hdf5"
painting_fullpath = paintingsDir + '/van_gogh-starry_night_over_the_rhone' + suffix 
y_styles = load_y_styles(painting_fullpath, style_layers_used)

print('Preparing training loss functions')
train_loss_style1_2 = frobenius_error(y_styles[0], grams(style_outputs_layer[0]))
train_loss_style2_2 = frobenius_error(y_styles[1], grams(style_outputs_layer[1]))
train_loss_style3_4 = frobenius_error(y_styles[2], grams(style_outputs_layer[2]))
train_loss_style4_4 = frobenius_error(y_styles[3], grams(style_outputs_layer[3]))
train_loss_style5_4 = frobenius_error(y_styles[4], grams(style_outputs_layer[4]))

reg_TV = total_variation_error(input_layer, 2)

print('Building white noise images')
input_data = preprocess(create_noise_tensor(height, width, channels)).transpose(0, 3, 1, 2)

current_iter = 1
for idx, feat_output in enumerate(feat_outputs_layer):
    if idx != 3:
        # conv_1_2 and conv_2_2  are too "clean" for human perception
        # conv_3_4 tends to be too "clean"
        # conv_5_4 layer doesn't hold enough information to rebuild the structure of the photo
        continue
    layer_name_feat = feat_layers_used[idx]
    train_loss_feat = squared_normalized_frobenius_error(train_feat_labels[idx], feat_output)
    print('Compiling VGG headless 5 for ' + layer_name_feat + ' feat reconstruction')
    for alpha in [1e2]:
        for beta in [5e0]:
            for gamma in [1e-3]:
                if alpha == beta and alpha != 1:
                    continue
                print("alpha, beta, gamma:", alpha, beta, gamma)

                print('Compiling model')
                # Based on previous analysis, the conv_2_2/conv_3_3 layers have
                train_loss = alpha * 0.25 * (train_loss_style1_2 + train_loss_style2_2 + train_loss_style3_4 + train_loss_style4_4 + train_loss_style5_4) \
                    + beta * train_loss_feat \
                    + gamma * reg_TV

                grads = K.gradients(train_loss, input_layer)[0]
                grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())
                train_iteratee = K.function([input_layer], [train_loss, grads, train_loss_style1_2, train_loss_style2_2, train_loss_style3_4, train_loss_style4_4, train_loss_style5_4, train_loss_feat])

                config = {'learning_rate': 1e-01}
                best_input_data, losses = train_input(input_data, train_iteratee, adam, config, max_iter=2000)

                prefix = str(current_iter).zfill(4)
                suffix = '_alpha' + str(alpha) +'_beta' + str(beta) + '_gamma' + str(gamma)
                fullOutPath = resultsDir + '/' + prefix + '_gatys_paper_feat' + layer_name_feat + suffix + '.png'
                dump_as_hdf5(resultsDir + '/' + prefix + '_gatys_paper_feat' + layer_name_feat + suffix + ".hdf5", best_input_data[0])
                save_image(fullOutPath, deprocess(best_input_data[0], dim_ordering='th'))
                plot_losses(losses, resultsDir, prefix, suffix)

                current_iter += 1