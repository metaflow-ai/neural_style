import os, json, gc

from keras import backend as K
from keras.engine.training import collect_trainable_weights
from keras.optimizers import Adam
# from keras.utils.visualize_util import plot

from vgg19.model_headless import VGG_19_headless_5, get_layer_data
from models.style_transfer import (style_transfer_conv_transpose,
                        style_transfer_upsample)

from utils.imutils import plot_losses, load_images, preprocess
from utils.lossutils import (grams, frobenius_error, 
                    train_weights, total_variation_error)

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
dataDir = dir + '/data'
resultsDir = dir + '/outpout'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)
trainDir = dataDir + '/train'
paintingsDir = dataDir + '/paintings'

channels = 3
width = 600
height = 600
input_shape = (channels, width, height)
batch_size = 4
max_number_of_epoch = 2

print('Loading style_transfer model')
stWeightsFullpath = dir + '/models/st_vangogh_weights.hdf5'
st_model = style_transfer_upsample(input_shape=input_shape)
if os.path.isfile(stWeightsFullpath): 
    print("Loading weights")
    st_model.load_weights(stWeightsFullpath)
init_weights = st_model.get_weights()

print('Loading painting')
X_train_style = load_images(dataDir + '/paintings', size=(height, width), dim_ordering='th', verbose=True)
X_train_style = X_train_style[7:8]
print("X_train_style shape:", X_train_style.shape)

print('Loading VGG headless 5')
modelWeights = vgg19Dir + '/vgg-19_headless_5_weights.hdf5'
vgg_model = VGG_19_headless_5(modelWeights, trainable=False)
layer_dict, layers_names = get_layer_data(vgg_model, 'conv_')
print('Layers found:' + ', '.join(layers_names))

print('Creating training labels')
style_layers_used = ['conv_1_2', 'conv_2_2', 'conv_3_4', 'conv_4_4']
style_outputs_layer = [grams(layer_dict[name].output) for name in style_layers_used]
predict_style = K.function([vgg_model.input], style_outputs_layer)
y_styles = predict_style([X_train_style])

[c11, c12, 
c21, c22, 
c31, c32, c33, c34, 
c41, c42, c43, c44,
c51, c52, c53, c54] = vgg_model(st_model.input)
y_feat = c42

print('Building full model')
[fm_c11, fm_c12, 
fm_c21, fm_c22, 
fm_c31, fm_c32, fm_c33, fm_c34,
fm_c41, fm_c42, fm_c43, fm_c44,
fm_c51, fm_c52, fm_c53, fm_c54] = vgg_model(st_model.output)
preds = [fm_c11, fm_c12, fm_c21, fm_c22, fm_c31, fm_c32, fm_c33, fm_c34, fm_c41, fm_c42, fm_c43, fm_c44, fm_c51, fm_c52, fm_c53, fm_c54]
pred_styles = [fm_c12, fm_c22, fm_c34, fm_c44]
pred_feat = fm_c42

print('Preparing training loss functions')
train_loss_styles = []
for idx, y_style in enumerate(y_styles):
    train_loss_styles.append(
        frobenius_error(
            y_style, 
            grams(pred_styles[idx])
        )
    )

train_loss_feat = frobenius_error(y_feat, pred_feat)

reg_TV = total_variation_error(st_model.output, 2)

print('Iterating over hyper parameters')
current_iter = 0
for alpha in [1e2]:
    for beta in [5.]:
        for gamma in [1e-03]:
            print("alpha, beta, gamma:", alpha, beta, gamma)

            gc.collect()
        
            st_model.set_weights(init_weights)
            print('Compiling train loss')
            tls = [train_loss_style * alpha * 1 / len(train_loss_styles) for train_loss_style in train_loss_styles]
            tlf = [train_loss_feat * beta]
            rtv = reg_TV * gamma
            train_loss =  sum(tls + tlf) + rtv

            print('Compiling Adam update')
            adam = Adam(lr=1e-03)
            updates = adam.get_updates(collect_trainable_weights(st_model), st_model.constraints, train_loss)

            print('Compiling train function')
            inputs = [st_model.input, K.learning_phase()]
            outputs = [train_loss] + tlf + tls
            train_iteratee = K.function(inputs, outputs, updates=updates)

            print('Starting training')
            best_trainable_weights, losses = train_weights(
                trainDir,
                (height, width),
                st_model, 
                train_iteratee, 
                cv_input_dir=None, 
                max_iter=1500,
                batch_size=batch_size
            )

            prefix = str(current_iter).zfill(4)
            suffix = '_alpha' + str(alpha) +'_beta' + str(beta) + '_gamma' + str(gamma)
            st_weights = resultsDir + '/' + prefix + 'st_vangogh_weights' + suffix + '.hdf5'
            fullpath_loss = resultsDir + '/' + prefix + 'st_vangogh_loss' + suffix + '.json'
            current_iter += 1

            print("Saving final data")
            st_model.set_weights(best_trainable_weights)
            st_model.save_weights(st_weights, overwrite=True)

            with open(fullpath_loss, 'w') as outfile:
                json.dump(losses, outfile)  

            plot_losses(losses, resultsDir, prefix, suffix)
