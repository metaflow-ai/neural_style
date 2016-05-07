import os
import numpy as np
import matplotlib.pyplot as plt
import json

from keras import backend as K
from keras.layers import Input
from keras.layers.core import Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
from keras.callbacks import ModelCheckpoint

from utils.lossutils import LossHistory, grams_frobenius_error, euclidian_error
from vgg16.model import VGG_16_mean 
from vgg16.model_headless import *
from models.style_transfer import style_transfer
from utils.imutils import load_images, load_image


dir = os.path.dirname(os.path.realpath(__file__))
vgg16Dir = dir + '/vgg16'
resultsDir = dir + '/models/results'
dataDir = dir + '/data'

############
# for 'th' dim_ordering (default): [batch, channels, height, width] 
############
print('Loading training set')
# X_train = load_images(dataDir + '/train')
X_train = load_images(dataDir + '/overfit')
print("X_train shape:", X_train.shape)

print('Loading Van Gogh')
vanGoghPath = dataDir + '/paintings/vangogh.jpg'
X_train_paint = np.array([load_image(vanGoghPath)])

print('Loading mean')
meanPath = vgg16Dir + '/vgg-16_mean.npy'
mean = VGG_16_mean(path=meanPath)

print('Loading VGG headless')
modelWeights = vgg16Dir + '/vgg-16_headless_5_weights.hdf5'
vgg_headless_model = VGG_16_headless_5(modelWeights, trainable=False)
print(np.array(vgg_headless_model.layers).shape)
# print(vgg_headless_model.summary())

print('Loading style_transfer')
stWeights = dir + '/models/st_vangogh_weights.hdf5'
if os.path.isfile(stWeights): 
    print("From weights")
    st_model = style_transfer(stWeights)
else:
    print("From scratch")
    st_model = style_transfer()
print(np.array(st_model.layers).shape)
# print(st_model.summary())

print('Building full model')
def get_output_shape(input_shape):
    return input_shape

input = Input(shape=(3, 256, 256), name='input')
st1 = st_model(input)
clip1 = Lambda(lambda x: K.clip(x, 0, 255), output_shape=get_output_shape)(st1)
l1 = Lambda(lambda x: x - mean, output_shape=get_output_shape)(clip1)
c11, c12, c21, c22, c31, c32, c33, c41, c42, c43, c51, c52, c53, c33_2 = vgg_headless_model(l1)
full_model = Model(input=[input], output=[
    c11, c12, 
    c21, c22, 
    c31, c32, c33, 
    c41, c42, c43,
    c51, c52, c53,
    c33_2]
)
# full_model.load_weights()

print('Ploting models')
plot(st_model, to_file=dir + '/models/st_model.png', show_shapes=True)
plot(vgg_headless_model, to_file=dir + '/models/vgg_headless_model.png', show_shapes=True)
plot(full_model, to_file=dir + '/models/full_model.png', show_shapes=True)

print('Creating labels')
vgg_headless_model.compile(loss='mean_squared_error', optimizer='sgd')
labels = []
(c11_plabels, c12_plabels, 
    c21_plabels, c22_plabels, 
    c31_plabels, c32_plabels, c33_plabels, 
    c41_plabels, c42_plabels, c43_plabels, 
    c51_plabels, c52_plabels, c53_plabels, 
    c33_2_plabels) = vgg_headless_model.predict(X_train_paint - mean)
for label in [c11_plabels, c12_plabels, 
        c21_plabels, c22_plabels, 
        c31_plabels, c32_plabels, c33_plabels, 
        c41_plabels, c42_plabels, c43_plabels, 
        c51_plabels, c52_plabels, c53_plabels]:
    labels.append(np.repeat(label, X_train.shape[0], axis=0))

(c11_ilabels, c12_ilabels, 
    c21_ilabels, c22_ilabels, 
    c31_ilabels, c32_ilabels, c33_ilabels, 
    c41_ilabels, c42_ilabels, c43_ilabels, 
    c51_ilabels, c52_ilabels, c53_ilabels, 
    c33_2_ilabels) = vgg_headless_model.predict(X_train - mean)
labels.append(c33_2_ilabels)
for label in labels:
    print(label.shape)

print('Compiling full_model')
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

full_model.compile(
    optimizer=adam,
    loss=[
        grams_frobenius_error, # conv_1_1
        grams_frobenius_error, # conv_1_2
        grams_frobenius_error, # conv_2_1
        grams_frobenius_error, # conv_2_2
        grams_frobenius_error, # conv_3_1
        grams_frobenius_error, # conv_3_2
        grams_frobenius_error, # conv_3_3
        grams_frobenius_error, # conv_4_1
        grams_frobenius_error, # conv_4_2
        grams_frobenius_error, # conv_4_3
        grams_frobenius_error, # conv_5_1
        grams_frobenius_error, # conv_5_2
        grams_frobenius_error, # conv_5_3
        euclidian_error # conv_3_3
    ],
    loss_weights=[ # Style is as important as feature
        0, # conv_1_1
        0.2, # conv_1_2
        0, # conv_2_1
        0.2, # conv_2_2
        0, # conv_3_1
        0, # conv_3_2
        0.2, # conv_3_3
        0, # conv_4_1
        0, # conv_4_2
        0.2, # conv_4_3
        0, # conv_5_1
        0, # conv_5_2
        0.2, # conv_5_3
        1, # conv_3_3
    ]
)


history = LossHistory()
fullpath_loss = resultsDir + '/full_model_losses.json'
if os.path.isfile(fullpath_loss): 
    print('Loading history')
    history.losses = json.load(fullpath_loss)

print('Training full_model')
full_model.fit(
    X_train,
    labels,
    nb_epoch=1, 
    batch_size=4,
    callbacks=[
        #ModelCheckpoint(
        #    resultsDir + '/full_model_weights.{epoch:02d}-{loss:.2f}.hdf5",
        #    monitor='loss',
        #    verbose=1
        #),
        history
    ]
)

print("Saving final data")
st_model.save_weights(stWeights, overwrite=True)

with open(fullpath_loss, 'w') as outfile:
    json.dump(history.losses, outfile)

plt.plot(history.losses)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.savefig(resultsDir + '/loss.png')
