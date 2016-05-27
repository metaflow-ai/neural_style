import os, json, h5py

from keras import backend as K
from keras.engine.training import collect_trainable_weights
from keras.layers.core import Lambda
from keras.optimizers import Adam
# from keras.utils.visualize_util import plot

from vgg19.model import VGG_19_mean 
from vgg19.model_headless import VGG_19_headless_5
from models.style_transfer import style_transfer

from utils.imutils import plot_losses
from utils.lossutils import (grams, frobenius_error, 
                    frobenius_error, train_weights,
                    total_variation_error)

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/vgg19'
resultsDir = dir + '/models/results/st'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)
dataDir = dir + '/data'
trainDir = dataDir + '/train'
paintingsDir = dataDir + '/paintings'

channels = 3
width = 256
height = 256
input_shape = (channels, width, height)
batch_size = 4
max_number_of_epoch = 2

print('Loading style_transfer model')
stWeightsFullpath = dir + '/models/st_vangogh_weights.hdf5'
st_model = style_transfer(input_shape=input_shape)
init_weights = st_model.get_weights()
if os.path.isfile(stWeightsFullpath): 
    print("Loading weights")
    st_model = st_model.load_weights(stWeightsFullpath)

print('Loading VGG headless 5 model')
modelWeights = vgg19Dir + '/vgg-16_headless_5_weights.hdf5'
vgg_model = VGG_19_headless_5(modelWeights, input_shape=input_shape, trainable=False, poolingType='average')
meanPath = vgg19Dir + '/vgg-16_mean.npy'
mean = VGG_19_mean(path=meanPath)

print('Loading label generator')
[c11, c12, 
c21, c22, 
c31, c32, c33, c34, 
c41, c42, c43, c44,
c51, c52, c53, c54] = vgg_model(st_model.input)
y_feat = c42

print('Building full model')
l_output = Lambda(lambda x: x - mean, output_shape=lambda shape: shape)(st_model.output)
[fm_c11, fm_c12, 
fm_c21, fm_c22, 
fm_c31, fm_c32, fm_c33, fm_c34,
fm_c41, fm_c42, fm_c43, fm_c44
fm_c51, fm_c52, fm_c53, fm_c54] = vgg_model(l_output)
preds = [fm_c11, fm_c12, fm_c21, fm_c22, fm_c31, fm_c32, fm_c33, fm_c34, fm_c41, fm_c42, fm_c43, fm_c44, fm_c51, fm_c52, fm_c53, fm_c54]
pred_style = [fm_c12, fm_c22, fm_c33, fm_c43]
pred_feat = fm_c33

print('Loading painting')
# suffix = "_ori.hdf5"
# suffix = "_600x600.hdf5"
suffix = "_256x256.hdf5"
painting_fullpath = paintingsDir + '/van_gogh-starry_night_over_the_rhone' + suffix 
with h5py.File(painting_fullpath, 'r') as f:
    y_styles = []
    y_styles.append(f['conv_1_2'][()])
    y_styles.append(f['conv_2_2'][()])
    y_styles.append(f['conv_3_3'][()])
    y_styles.append(f['conv_4_3'][()])
    y_styles.append(f['conv_5_3'][()])


print('preparing loss functions')
loss_style1_2 = frobenius_error(y_styles[0], grams(pred_style[0]))
loss_style2_2 = frobenius_error(y_styles[1], grams(pred_style[1]))
loss_style3_3 = frobenius_error(y_styles[2], grams(pred_style[2]))
loss_style4_3 = frobenius_error(y_styles[3], grams(pred_style[3]))
train_loss_feat = frobenius_error(y_feat, pred_feat)
reg_TV = total_variation_error(l_output)

print('Iterating over hyper parameters')
current_iter = 0
for alpha in [1e-03, 1e-04]:
    for beta in [1.]:
        for gamma in [1e-04, 1e-05]:
            print("alpha, beta, gamma:", alpha, beta, gamma)

            st_model.set_weights(init_weights)
            print('Compiling train loss')
            train_loss = alpha * 0.25 * (loss_style1_2 + loss_style2_2 + loss_style3_3 + 1e03 * loss_style4_3) \
                + beta * train_loss_feat \
                + gamma * reg_TV

            print('Compiling Adam update')
            adam = Adam(lr=1e-03)
            updates = adam.get_updates(collect_trainable_weights(st_model), st_model.constraints, train_loss)

            print('Compiling train function')
            train_iteratee = K.function([st_model.input, K.learning_phase()], [train_loss], updates=updates)

            print('Starting training')
# X_train = load_images(trainDir, size=(height, width))
# print("X_train shape: " + str(X_train.shape))


# print('Loading cross validation images')
# # X_cv = load_images(dataDir + '/val')
# X_cv = load_images(dataDir + '/overfit/cv', size=(height, width))
# print("X_cv shape: " + str(X_cv.shape))


#             if len(X_cv):
#                 print('Preparing cv iteratee function')
#                 cv_loss = alpha * 0.25 * (loss_style1_2 + loss_style2_2 + loss_style3_3 + 1e03 * loss_style4_3) \
#                     + beta * cv_loss_feat \
#                     + gamma * reg_TV
#                 cross_val_iteratee = K.function([st_model.input, K.learning_phase()], [cv_loss])
#             else:
#                 cross_val_iteratee = None

            best_trainable_weights, losses = train_weights(
                trainDir,
                (height, width),
                st_model, 
                train_iteratee, 
                cv_input_dir=None, 
                max_iter=1500,
                batch_size=4
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
