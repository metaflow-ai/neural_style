import os, json, time, argparse

from keras import backend as K
from keras.optimizers import Adam
# from keras.utils.visualize_util import plot as plot_model

from models.style_transfer import style_transfer_conv_transpose, style_transfer_conv_inception_3

from utils.imutils import plot_losses, load_images, load_data
from utils.lossutils import (frobenius_error)
from utils.general import export_model

if K._BACKEND == "tensorflow":
    K.set_image_dim_ordering('tf')
else:
    K.set_image_dim_ordering('th')

dir = os.path.dirname(os.path.realpath(__file__))
dataDir = dir + '/data'
trainDir = dataDir + '/train'
overfitDir = dataDir + '/overfit'

parser = argparse.ArgumentParser(
    description='Neural artistic style. Generates an image by combining '
                'the content of an image and the style of another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--training_mode', default='identity', type=str, choices=['identity', 'overfit'], help='Load pretrained weights')
parser.add_argument('--weights', default='', type=str, help='Load pretrained weights')
parser.add_argument('--batch_size', default=8, type=int, help='batch size.')
parser.add_argument('--image_size', default=600, type=int, help='Input image size.')
parser.add_argument('--nb_epoch', default=1000, type=int, help='Number of epoch.')
parser.add_argument('--nb_res_layer', default=6, type=int, help='Number of residual layers in the style transfer model.')
args = parser.parse_args()

results_dir = dir + '/models/data/' + args.training_mode
if not os.path.isdir(results_dir): 
    os.makedirs(results_dir)

dim_ordering = K.image_dim_ordering()
channels = 3
width = args.image_size
height = args.image_size
size = (height, width)
if dim_ordering == 'th':
    input_shape = (channels, width, height)
else:
    input_shape = (width, height, channels)
batch_size = args.batch_size

print('Loading data')
if args.training_mode == 'identity':
    X = load_images(overfitDir, size=(height, width), verbose=False, st=True)
    y = X.copy()
    X_cv = load_images(overfitDir + '/cv', size=(height, width), verbose=False, st=True)
    y_cv = X_cv.copy()
elif args.training_mode == 'overfit':
    (X, y), (X_cv, y_cv) = load_data(overfitDir, size=(height, width), verbose=False, st=True)
else:
    raise Exception('training_mode unknown: %s' % args.training_mode)
print('X.shape', X.shape)
print('y.shape', y.shape)
print('X_cv.shape', X_cv.shape)
print('y_cv.shape', y_cv.shape)

print('Loading model')
st_model = style_transfer_conv_inception_3(input_shape, mode=2, nb_res_layer=args.nb_res_layer) # th ordering, BGR
if os.path.isfile(args.weights): 
    print("Loading weights")
    st_model.load_weights(args.weights)

print('Compiling model')
adam = Adam(lr=1e-3)
st_model.compile(adam, loss='mse') # loss=frobenius_error (this is not giving the same loss)

print('Training model')
history = st_model.fit(X, y, batch_size=args.batch_size, nb_epoch=args.nb_epoch, verbose=1, validation_data=(X_cv, y_cv))
losses = history.history

print("Saving final data")
prefixedDir = prefixed_dir = "%s/%s-%s-%s" % (results_dir, str(int(time.time())), K._BACKEND, dim_ordering)
export_model(st_model, prefixedDir)
with open(prefixedDir + '/losses.json', 'w') as outfile:
    json.dump(losses, outfile)  
plot_losses(losses, prefixedDir)