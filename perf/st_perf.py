import os, sys, time

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

from keras.utils.visualize_util import plot as plot_model
from models.style_transfer import style_transfer_conv_transpose as st
 
from utils.imutils import load_images

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/../vgg19'
resultsDir = dir + '/../models/results/st2'
dataDir = dir + '/../data'
testDir = dataDir + '/test'

channels = 3
width = 600
height = 600
input_shape = (channels, width, height)
batch = 4

print('Loading test set')
X_test = load_images(testDir, limit=4, size=(height, width), dim_ordering='th', st=True)
print('X_test.shape: ' + str(X_test.shape))


model = st(input_shape=input_shape)
plot_model(model, to_file=dir + '/model.png', show_shapes=True)
total_params = 0
for i in range(len(model.layers)):
    total_params += model.layers[i].count_params()
print('Total number of params:' + str(total_params))

print('Predicting')
# time it
start = time.clock()
num_loop = 1
for i in range(num_loop):
    results = model.predict(X_test) # Equivalent to predict([X_test, False])
end = time.clock()
duration_batch = (end-start)/(X_test.shape[0] * num_loop)

start = time.clock()
num_loop = 1
for i in range(X_test.shape[0]):
    results = model.predict(X_test[i:i+1, :, :, :]) # Equivalent to predict([X_test, False])
end = time.clock()
duration = (end-start)/(X_test.shape[0] * num_loop)

print("duration taken on 1 average call when batching: " + str(duration_batch))
print("duration taken on 1 average call when looping: " + str(duration))