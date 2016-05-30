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


channels = 3
width = 256
height = 256
input_shape = (channels, width, height)
batch = 4

print('Loading a cat image')
X_train = load_images(dataDir + '/overfit', size=(height, width), limit=1, dim_ordering='th', verbose=True)
print("X_train shape:", X_train.shape)

print('Loading painting')
X_train_style = load_images(dataDir + '/paintings', size=(height, width), limit=1, dim_ordering='th', verbose=True)
print("X_train_style shape:", X_train_style.shape)

print('Loading VGG headless 5')
modelWeights = vgg19Dir + '/vgg-19_headless_5_weights.hdf5'
model = VGG_19_headless_5(modelWeights, trainable=False)
layer_dict, layers_names = get_layer_data(model, 'conv_')

input_layer = model.input

print('Building white noise images')
input_data = create_noise_tensor(height, width, channels).transpose(0, 3, 1, 2).astype(K.floatx())

print('Using optimizer: ' + optimizer)
current_iter = 1
for idx_feat, layer_name_feat in enumerate(layers_names):
    for idx_style, layer_name_style in enumerate(layers_names):
        print('Creating labels for feat ' + layer_name_feat + ' and style ' + layer_name_style)
        out_style = layer_dict[layer_name_style].output
        predict_style = K.function([input_layer], out_style)
        out_style_labels = predict_style([X_train_style])

        out_feat = layer_dict[layer_name_feat].output
        predict_feat = K.function([input_layer], out_feat)
        out_feat_labels = predict_feat([X_train])

        loss_style = frobenius_error(grams(out_style_labels), grams(out_style))
        loss_feat = frobenius_error(out_feat_labels, out_feat)
        reg_TV = total_variation_error(input_layer, 2)

        print('Compiling VGG headless 5 for feat ' + layer_name_feat + ' and style ' + layer_name_style)
        for alpha in [1e2]:
            for beta in [5.]:
                for gamma in [1e-3]:
                    if alpha == beta and alpha != 1:
                        continue
                    print("alpha, beta, gamma:", alpha, beta, gamma)

                    print('Compiling model')
                    ls = alpha * loss_style
                    lf = beta * loss_feat
                    rtv = gamma * reg_TV
                    loss =  ls + lf + rtv
                    grads = K.gradients(loss, input_layer)
                    if optimizer == 'adam':
                        grads = norm_l2(grads)
                    iterate = K.function([input_layer], [loss, grads, lf, ls])

                    config = {'learning_rate': 5e-00}
                    best_input_data, losses = train_input(input_data, iterate, optimizer, config, max_iter=1000)

                    prefix = str(current_iter).zfill(4)
                    suffix = '_alpha' + str(alpha) +'_beta' + str(beta) + '_gamma' + str(gamma)
                    fullOutPath = resultsDir + '/' + prefix + '_gatys_st' + layer_name_style + '_feat' + layer_name_feat + suffix + ".png"
                    dump_as_hdf5(resultsDir + '/' + prefix + '_gatys_st' + layer_name_style + '_feat' + layer_name_feat + suffix + ".hdf5", best_input_data[0])
                    save_image(fullOutPath, deprocess(best_input_data[0], dim_ordering='th'))
                    plot_losses(losses, resultsDir, prefix, suffix)

                    current_iter += 1
