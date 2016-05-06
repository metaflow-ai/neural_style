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

############
# for 'th' dim_ordering (default): [batch, channels, height, width] 
############
print('Loading training set')
trainPath = dir + '/data/train'
X_train = load_images(trainPath, 4)
print("X_train shape:", X_train.shape)

print('Loading Van Gogh')
vanGoghPath = dir + '/data/paintings/vangogh.jpg'
X_train_paint = np.array([load_image(vanGoghPath)])

print('Loading mean')
meanPath = vgg16Dir + '/vgg-16_mean.npy'
mean = VGG_16_mean(path=meanPath)

print('Loading VGG headless')
modelWeights = vgg16Dir + '/vgg-16_headless_5_weights.hdf5'
vgg_headless_model = VGG_16_headless(modelWeights, trainable=False)
print(np.array(vgg_headless_model.layers).shape)
# print(vgg_headless_model.summary())

print('Loading style_transfer')
stWeights = dir + '/models/st_vangogh_weights.hdf5'
if os.path.isfile(stWeights): 
    st_model = style_transfer(stWeights)
else:
    st_model = style_transfer()
print(np.array(st_model.layers).shape)
# # print(st_model.summary())

print('Building full model')
def get_output_shape(input_shape):
    return input_shape

input = Input(shape=(3, 256, 256), name='input')
st1 = st_model(input)
clip1 = Lambda(lambda x: K.clip(x, 0, 255), output_shape=get_output_shape)(st1)
l1 = Lambda(lambda x: x - mean, output_shape=get_output_shape)(clip1)
out_style1, out_style2, out_style3, out_feat3, out_style4, out_style5 = vgg_headless_model(l1)
full_model = Model(input=[input], output=[out_style1, out_style2, out_style3, out_feat3, out_style4, out_style5])
# full_model.load_weights()

# print('Ploting models')
# plot(st_model, show_shapes=True)
# plot(full_model, show_shapes=True)
# plot(full_model, show_shapes=True)

print('Compiling vgg_headless_model')
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

vgg_headless_model.compile(
    optimizer=adam,
    loss={
        'out_style1': grams_frobenius_error,
        'out_style2': grams_frobenius_error,
        'out_style3': grams_frobenius_error,
        'out_feat3': euclidian_error,
        'out_style4': grams_frobenius_error,
        'out_style5': grams_frobenius_error,
    }
)

print('Creating labels')
plabels_style1, plabels_style2, plabels_style3, plabels_feat3, plabels_style4, plabels_style5 = vgg_headless_model.predict(X_train_paint)
plabels_style1 = np.repeat(plabels_style1, X_train.shape[0], axis=0)
plabels_style2 = np.repeat(plabels_style2, X_train.shape[0], axis=0)
plabels_style3 = np.repeat(plabels_style3, X_train.shape[0], axis=0)
plabels_feat3 = np.repeat(plabels_feat3, X_train.shape[0], axis=0)
plabels_style4 = np.repeat(plabels_style4, X_train.shape[0], axis=0)
plabels_style5 = np.repeat(plabels_style5, X_train.shape[0], axis=0)
print(plabels_style1.shape, plabels_style2.shape, plabels_style3.shape, plabels_feat3.shape, plabels_style4.shape, plabels_style5.shape)

ilabels_style1, ilabels_style2, ilabels_style3, ilabels_feat3, ilabels_style4, ilabels_style5 = vgg_headless_model.predict(X_train)
print(ilabels_style1.shape, ilabels_style2.shape, ilabels_style3.shape, ilabels_feat3.shape, ilabels_style4.shape, ilabels_style5.shape)

print('Compiling full_model')
full_model.compile(
    optimizer=adam,
    loss=[
        grams_frobenius_error,
        grams_frobenius_error,
        grams_frobenius_error,
        euclidian_error,
        grams_frobenius_error,
        grams_frobenius_error,
    ]
)

print('Training full_model')
lossFile = dir + '/models/results/full_model_losses.txt'
history = LossHistory()
fullpath_loss = dir + '/models/results/full_model_losses.txt'
# with open('data.txt') as infile:
#     history.losses = json.load(infile)

full_model.fit(
    X_train,
    [
        plabels_style1,
        plabels_style2,
        plabels_style3,
        ilabels_feat3,
        plabels_style4,
        plabels_style5,
    ],
    nb_epoch=2, 
    batch_size=4,
    callbacks=[
        ModelCheckpoint(
            dir + "/models/results/full_model_weights.{epoch:02d}-{loss:.2f}.hdf5",
            monitor='loss',
            verbose=1
        ),
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
plt.savefig(dir + '/models/results/loss.png')
