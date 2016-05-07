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

print('Loading a random image')
X_train = load_images(dataDir + '/overfit', 1)
print("X_train shape:", X_train.shape)

print('Loading Van Gogh')
vanGoghPath = dataDir + '/paintings/vangogh.jpg'
X_train_paint = np.array([load_image(vanGoghPath)])

print('Building white noise image')
white_im = numpy.random.rand(3, 256, 256) * 255

print('Loading mean')
meanPath = vgg16Dir + '/vgg-16_mean.npy'
mean = VGG_16_mean(path=meanPath)

print('Loading VGG headless 5')
modelWeights = vgg16Dir + '/vgg-16_headless_5_weights.hdf5'
model = VGG_16_headless_5(modelWeights, trainable=False)
print(np.array(model.layers).shape)
# print(model.summary())

print('Creating labels')
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(
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
labels = []
(c11_plabels, c12_plabels, 
    c21_plabels, c22_plabels, 
    c31_plabels, c32_plabels, c33_plabels, 
    c41_plabels, c42_plabels, c43_plabels, 
    c51_plabels, c52_plabels, c53_plabels, 
    c33_2_plabels) = model.predict(X_train_paint - mean)
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
    c33_2_ilabels) = model.predict(X_train - mean)
labels.append(c33_2_ilabels)
for label in labels:
    print(label.shape)


print('Training full_model')
model.fit(
    X_train,
    labels,
    nb_epoch=1, 
    batch_size=4,
    callbacks=[]
)

print("Saving final data")
st_model.save_weights(stWeights, overwrite=True)

with open(fullpath_loss, 'w') as outfile:
    json.dump(history.losses, outfile)

plt.plot(history.losses)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.savefig(resultsDir + '/loss.png')
