import os

from keras import backend as K

from vgg19.model_headless import VGG_19_headless_5, get_layer_data

from utils.imutils import *
from utils.lossutils import *

optimizer = 'adam'
if optimizer == 'lbfgs':
    K.set_floatx('float64') # scipy needs float64 to use lbfgs

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
dataDir = dir + '/data'
resultsDir = dataDir + '/output/vgg19'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)


channels = 3
width = 256
height = 256
input_shape = (channels, width, height)
batch = 4

# data = load_hdf5_im(dataDir + '/output/0001_gatys_paper_featconv_4_1_alpha100.0_beta5.0_gamma0.001.hdf5')
# save_image(dir + '/models/results/vgg19/a.png', deprocess(data, dim_ordering='th'))
# save_image(dir + '/models/results/vgg19/b.png', deprocess(data, dim_ordering='th', normalize=True))

print('Loading a cat image')
X_train = load_images(dataDir + '/overfit', size=(height, width), limit=1, dim_ordering='th')
print("X_train shape:", X_train.shape)

print('Loading painting')
X_train_style = load_images(dataDir + '/paintings', size=(height, width), limit=1, dim_ordering='th')
print("X_train_style shape:", X_train_style.shape)

print('Loading VGG headless 5')
modelWeights = vgg19Dir + '/vgg-19_headless_5_weights.hdf5'
model = VGG_19_headless_5(modelWeights, trainable=False)
layer_dict, layers_names = get_layer_data(model, 'conv_')
print('Layers found:' + ', '.join(layers_names))

input_layer = model.input

print('Building white noise images')
input_data = create_noise_tensor(height, width, channels).transpose(0, 3, 1, 2).astype(K.floatx())

print('Using optimizer: ' + optimizer)
current_iter = 1
for layer_name in layers_names:
    print('Creating labels for ' + layer_name)
    out = layer_dict[layer_name].output
    predict = K.function([input_layer], [out])

    y_style = predict([X_train_style])[0].copy()
    y_feat = predict([X_train])[0].copy()
    
    reg_TV = total_variation_error(input_layer, 2)

    for gamma in [1e-05, 0]:
        print('gamma:' + str(gamma))
        print('Compiling VGG headless 1 for ' + layer_name + ' style reconstruction')
        loss_style = frobenius_error(grams(y_style).copy(), grams(out).copy())
        loss_TV =  gamma * reg_TV
        total_loss_style = loss_style + loss_TV
        grads_style = K.gradients(total_loss_style, input_layer)
        if optimizer == 'adam':
            grads_style /= (K.sqrt(K.mean(K.square(grads_style))) + K.epsilon())
        iterate_style = K.function([input_layer], [total_loss_style, grads_style])

        print('Compiling VGG headless 1 for ' + layer_name + ' feature reconstruction')
        loss_feat = frobenius_error(y_feat, out)
        loss_TV =  gamma * reg_TV
        total_loss_feat = loss_feat + loss_TV
        grads_feat = K.gradients(total_loss_feat, input_layer)
        if optimizer == 'adam':
            grads_feat /= (K.sqrt(K.mean(K.square(grads_feat))) + K.epsilon())
        iterate_feat = K.function([input_layer], [total_loss_feat, grads_feat])

        prefix = str(current_iter).zfill(4)
        suffix = '_gamma' + str(gamma)

        print('Training the image for style')
        config = {'learning_rate': 5e-1}
        best_input_style_data, style_losses = train_input(input_data, iterate_style, optimizer, config, max_iter=1500)
        fullOutPath = resultsDir + '/' + prefix + '_style_' + layer_name + suffix + ".png"
        dump_as_hdf5(resultsDir + '/' + prefix + '_style_' + layer_name + suffix + ".hdf5", best_input_style_data[0])
        save_image(fullOutPath, deprocess(best_input_style_data[0], dim_ordering='th'))
        plot_losses(style_losses, resultsDir, prefix + '_style', suffix)

        print('Training the image for feature')
        config = {'learning_rate': 5e-1}
        best_input_feat_data, feat_losses = train_input(input_data, iterate_feat, optimizer, config, max_iter=1500)
        fullOutPath = resultsDir + '/' + prefix + '_feat_' + layer_name + suffix + ".png"
        dump_as_hdf5(resultsDir + '/' + prefix + '_feat_' + layer_name + suffix + ".hdf5", best_input_feat_data[0])
        save_image(fullOutPath, deprocess(best_input_feat_data[0], dim_ordering='th'))
        plot_losses(feat_losses, resultsDir, prefix + '_feat', suffix)

        current_iter += 1