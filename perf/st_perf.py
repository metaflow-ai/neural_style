from __future__ import absolute_import

import os, sys, time, argparse, re
import numpy as np

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

from keras import backend as K 

from utils.imutils import load_images, resize
from utils.general import import_model
from models.layers import custom_objects

if K._BACKEND == "tensorflow":
    K.set_image_dim_ordering('tf')
else:
    K.set_image_dim_ordering('th')

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

width = args.image_size
height = args.image_size

X_test = load_images(test_dir, limit=args.batch_size, size=(height, width), preprocess_type='st', verbose=True)
print('X_test.shape: ' + str(X_test.shape))

current_iter = 0
subdirs = [x[0] for x in os.walk(args.models_dir)]
subdirs.pop(0) # First element is the parent dir
for idx, absolute_model_dir in enumerate(subdirs):
    print('Loading model in %s' % absolute_model_dir)
    st_model = import_model(absolute_model_dir, best=True, 
        should_convert=False, custom_objects=custom_objects)

    if not len(re.findall('superresolution', absolute_model_dir)):
        print('Timing batching')
        start = time.clock()
        results = st_model.predict(X_test)
        end = time.clock()
        duration_batch = (end-start)/X_test.shape[0]

    print('Timing looping')
    start = time.clock()
    num_loop = X_test.shape[0]
    for i in range(num_loop):
        if len(re.findall('superresolution', absolute_model_dir)):
            im = resize(X_test[0], (args.image_size/4, args.image_size/4))
        else:
            im = X_test[0]
        results = st_model.predict(np.array([im]))
    end = time.clock()
    duration_loop = (end-start)/num_loop

    print("duration taken on 1 average call when batching: " + str(duration_batch))
    print("duration taken on 1 average call when looping: " + str(duration_loop))