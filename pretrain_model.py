import os, json, time, argparse

from keras import backend as K
from keras.optimizers import Adam
# from keras.utils.visualize_util import plot as plot_model

from models.style_transfer import style_transfer_conv_transpose, style_transfer_conv_inception_3

from utils.imutils import plot_losses, load_images, load_data
from utils.lossutils import (frobenius_error)
from utils.general import export_model

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

resultsDir = dir + '/models/data/' + args.training_mode
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)

channels = 3
width = args.image_size
height = args.image_size
input_shape = (channels, width, height)
batch_size = args.batch_size

print('Loading data')
if args.training_mode == 'identity':
    X = load_images(overfitDir, size=(width, height), dim_ordering='th', verbose=False, st=True)
    y = X.copy()
    X_cv = load_images(overfitDir + '/cv', size=(width, height), dim_ordering='th', verbose=False, st=True)
    y_cv = X_cv.copy()
elif args.training_mode == 'overfit':
    (X, y), (X_cv, y_cv) = load_data(overfitDir, size=(width, height), dim_ordering='th', verbose=False, st=True)
else:
    raise Exception('training_mode unknown: %s' % args.training_mode)


st_model = style_transfer_conv_inception_3(input_shape=input_shape, nb_res_layer=args.nb_res_layer) # th ordering, BGR
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
prefixedDir = resultsDir + '/' + str(int(time.time()))
export_model(st_model, prefixedDir)
with open(prefixedDir + '/losses.json', 'w') as outfile:
    json.dump(losses, outfile)  
plot_losses(losses, prefixedDir)