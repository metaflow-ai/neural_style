import numpy as np
from scipy import misc, ndimage as ndi
from model import VGG_16, VGG_16_mean 
from model_mo import VGG_16_MO
import os, re
from keras.optimizers import Adam

dir = os.path.dirname(os.path.realpath(__file__))

# filename = dir + '/data/12.jpg'
# im = ndi.imread(filename, mode='RGB')
# im = misc.imresize(im, (224,224)).transpose(2, 0, 1).astype(np.float32)


# print('Loading images')
# testPath = dir + '/../data'
# X_test = []
# files = [f for f in os.listdir(testPath) if len(re.findall('3621588066_1c5fae5e2d.jpg$', f))]
# for filename in files:
#     fullpath = testPath + '/' + filename
#     im = ndi.imread(fullpath, mode='RGB')
#     im = misc.imresize(im, (224,224)).transpose(2, 0, 1).astype(np.float32)
#     X_test.append(im)


#     break
# print(np.array(X_test).shape)

# print('Loading mean')
# meanPath = dir + '/vgg-16_mean.npy'
# mean = VGG_16_mean(path=meanPath)
# print(mean.shape)

# print('Loading VGG')
# modelWeights = dir + '/vgg-16_weights.h5'
# model = VGG_16(modelWeights)
# print(np.array(model.layers).shape)
# # print(model.summary())

# print('Loading VGG MO')
# modelMoWeights = dir + '/vgg-16_mo_weights.h5'
# model_mo = VGG_16_MO(modelMoWeights)
# print(np.array(model_mo.layers).shape)
# print(len(model_mo.layers))
# # print(model_mo.summary())

# print('Compiling model')
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# model.compile(loss='categorical_crossentropy', optimizer=adam)

# print('Compiling model_mo')
# model_mo.compile(loss='categorical_crossentropy', optimizer=adam)

# print('Predicting classes')
# X_test -= mean
# classes = model.predict(X_test)
# print(classes.shape)
# print(np.argmax(classes, axis=1))

# print('Predicting mo classes')
# out1, out2, out3, out4, out5, classes = model_mo.predict(X_test)
# print(np.argmax(classes, axis=1))




from utils.copy_seq_weights import copySeqWeights
from model_headless import VGG_16_headless

model = VGG_16_headless()
copySeqWeights(model, dir + '/vgg-16_weights.h5', dir + '/vgg-16_headless_weights.h5', limit=30)
