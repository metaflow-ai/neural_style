from __future__ import absolute_import

import os, sys, time, argparse

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')
 
from keras.utils.visualize_util import plot as plot_model

from utils.imutils import load_images
from utils.general import import_model
from models.layers.ConvolutionTranspose2D import ConvolutionTranspose2D
from models.layers.ScaledSigmoid import ScaledSigmoid

dir = os.path.dirname(os.path.realpath(__file__)) + '/..'
dataDir = dir + '/data'
output_dir = dataDir + '/output'
test_dir = dataDir + '/test'

parser = argparse.ArgumentParser(
    description='Neural artistic style. Generates an image by combining '
                'the content of an image and the style of another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--models_dir', default=dir + '/models/data/st', type=str, help='Models top directories.')
parser.add_argument('--batch_size', default=30, type=int, help='batch size.')
parser.add_argument('--image_size', default=600, type=int, help='Input image size.')
args = parser.parse_args()

channels = 3
width = args.image_size
height = args.image_size
input_shape = (channels, width, height)

X_test = load_images(test_dir, limit=args.batch_size, size=(height, width), verbose=True, st=True)
print('X_test.shape: ' + str(X_test.shape))

current_iter = 0
subdirs = [x[0] for x in os.walk(args.models_dir)]
subdirs.pop(0) # First element is the parent dir
for idx, absolute_model_dir in enumerate(subdirs):
    print('Loading model in %s' % absolute_model_dir)
    st_model = import_model(absolute_model_dir, best=True, should_convert=False, custom_objects={
        'ConvolutionTranspose2D': ConvolutionTranspose2D,
        'ScaledSigmoid': ScaledSigmoid
    })
    plot_model(st_model, to_file=output_dir + '/' + str(idx) + '.png', show_shapes=True)

    print('Timing batching')
    start = time.clock()
    results = st_model.predict(X_test)
    end = time.clock()
    duration_batch = (end-start)/X_test.shape[0]

    print('Timing looping')
    start = time.clock()
    num_loop = 30
    for i in range(num_loop):
        results = st_model.predict(X_test[0:1, :, :, :])
    end = time.clock()
    duration_loop = (end-start)/num_loop

    print("duration taken on 1 average call when batching: " + str(duration_batch))
    print("duration taken on 1 average call when looping: " + str(duration_loop))