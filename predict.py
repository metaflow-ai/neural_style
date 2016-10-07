from __future__ import absolute_import

import os, argparse, time

from keras import backend as K

from utils.imutils import load_images, save_image
from utils.general import import_model
from models.layers import custom_objects

if K._BACKEND == "tensorflow":
    K.set_image_dim_ordering('tf')
else:
    K.set_image_dim_ordering('th')
    
dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    description='Neural artistic style. Generates an image by combining '
                'the content of an image and the style of another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--models_dir', default=dir + '/models/data/st', type=str, help='Models top directories.')
parser.add_argument('--batch_size', default=20, type=int, help='batch size.')
parser.add_argument('--image_size', default=600, type=int, help='Input image size.')
args = parser.parse_args()

dataDir = dir + '/data'
output_dir = dataDir + '/output'
overfit_dir = dataDir + '/overfit_' + str(args.image_size)
test_dir = dataDir + '/test'

dim_ordering = K.image_dim_ordering()
channels = 3
width = args.image_size
height = args.image_size
if dim_ordering == 'th':
    input_shape = (channels, width, height)
else:
    input_shape = (width, height, channels)

X_overfit = load_images(overfit_dir, limit=args.batch_size, size=(height, width), preprocess_type='st', verbose=True)
X_test = load_images(test_dir, limit=args.batch_size, size=(height, width), preprocess_type='st', verbose=True)
print('X_test.shape: ' + str(X_test.shape))
print('X_overfit.shape: ' + str(X_overfit.shape))

current_iter = 0
subdirs = [x[0] for x in os.walk(args.models_dir)]
subdirs.pop(0) # First element is the parent dir
for absolute_model_dir in subdirs:    
    print('Loading model in %s' % absolute_model_dir)
    st_model = import_model(absolute_model_dir, best=True, 
                    should_convert=False, custom_objects=custom_objects)

    print('Predicting')
    start = time.clock()
    results = st_model.predict(X_test) # Equivalent to predict([X_test, False])
    results_overfit = st_model.predict(X_overfit) # Equivalent to predict([X_test, False])
    end = time.clock()
    duration_batch = (end-start)/(X_test.shape[0] * X_overfit.shape[0])

    print('Dumping results')
    tmp_output_dir = output_dir + '/' + absolute_model_dir.split('/')[-2] + '/' + absolute_model_dir.split('/')[-1]
    if not os.path.isdir(tmp_output_dir): 
        os.makedirs(tmp_output_dir)
    for idx, im in enumerate(results):
        prefix = str(current_iter).zfill(4)
        fullOutPath = tmp_output_dir + '/' + prefix + "_" + str(idx) + ".png"
        save_image(fullOutPath, im, deprocess_type='st')
        fullOriPath = tmp_output_dir + '/' + prefix + "_" + str(idx) + "_ori.png"
        save_image(fullOriPath, X_test[idx], deprocess_type='st')

        current_iter += 1

    for idx, im in enumerate(results_overfit):
        prefix = str(current_iter).zfill(4)
        fullOutPath = tmp_output_dir + '/' + prefix + str(idx) + "_overfit.png"
        save_image(fullOutPath, im, deprocess_type='st')
        fullOriPath = tmp_output_dir + '/' + prefix + str(idx) + "_overfit_ori.png"
        save_image(fullOriPath, X_overfit[idx], deprocess_type='st')

        current_iter += 1
