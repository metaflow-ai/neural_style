import os, json, time, argparse

from keras import backend as K
from keras.optimizers import Adam
# from keras.utils.visualize_util import plot as plot_model

from models.style_transfer import (st_convt, st_conv_inception, st_convt_inception_prelu,
                        st_conv_inception_4, st_conv_inception_4_fast,
                        st_conv_inception_4_superresolution)

from utils.imutils import plot_losses, load_images, load_data, resize
from utils.general import export_model

if K._BACKEND == "tensorflow":
    K.set_image_dim_ordering('tf')
else:
    K.set_image_dim_ordering('th')

dir = os.path.dirname(os.path.realpath(__file__))
dataDir = dir + '/data'
trainDir = dataDir + '/train'
overfitDir = dataDir + '/overfit_600'

parser = argparse.ArgumentParser(
    description='Neural artistic style. Generates an image by combining '
                'the content of an image and the style of another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--model', default='inception', type=str, choices=['transpose', 'inception', 'inception_prelu', 'inception_4', 'inception_4_fast', 'superresolution'], help='Load pretrained weights')
parser.add_argument('--training_mode', default='identity', type=str, choices=['identity', 'overfit'], help='Load pretrained weights')
parser.add_argument('--weights', default='', type=str, help='Load pretrained weights')
parser.add_argument('--batch_size', default=4, type=int, help='batch size.')
parser.add_argument('--image_size', default=600, type=int, help='Input image size.')
parser.add_argument('--nb_epoch', default=30, type=int, help='Number of epoch.')
parser.add_argument('--nb_res_layer', default=6, type=int, help='Number of residual layers in the style transfer model.')
parser.add_argument('--lr', default=1e-3, type=float, help='The learning rate')
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
    X = load_images(overfitDir, size=(height, width), preprocess_type='st', verbose=False)
    y = X.copy()
    X_cv = load_images(overfitDir + '/cv', size=(height, width), preprocess_type='st', verbose=False)
    y_cv = X_cv.copy()
elif args.training_mode == 'overfit':
    (X, y), (X_cv, y_cv) = load_data(overfitDir, size=(height, width), preprocess_type='st', verbose=False)
else:
    raise Exception('training_mode unknown: %s' % args.training_mode)
print('X.shape', X.shape)
print('y.shape', y.shape)
print('X_cv.shape', X_cv.shape)
print('y_cv.shape', y_cv.shape)

print('Loading model')
# mode 1 should be possible but keras is complaining in the train.py file
# it doesn't like the idea that i will call later `output = pretrain_model(input)`
if args.model == 'transpose':
    st_model = st_convt(input_shape, mode=2, nb_res_layer=args.nb_res_layer)
elif args.model == 'inception':
    st_model = st_conv_inception(input_shape, mode=2, nb_res_layer=args.nb_res_layer)
elif args.model == 'inception_prelu':
    st_model = st_convt_inception_prelu(input_shape, mode=2, nb_res_layer=args.nb_res_layer)
elif args.model == 'inception_4':
    st_model = st_conv_inception_4(input_shape, mode=2, nb_res_layer=args.nb_res_layer)
elif args.model == 'inception_4_fast':
    st_model = st_conv_inception_4_fast(input_shape, mode=2, nb_res_layer=args.nb_res_layer)
elif args.model == 'superresolution':
    X = resize(X, (height/4, width/4))
    X_cv = resize(X_cv, (height/4, width/4))
    st_model = st_conv_inception_4_superresolution(input_shape, mode=1, nb_res_layer=args.nb_res_layer)
else:
    raise Exception('Model name %s not allowed , should not happen anyway' % args.model)

if os.path.isfile(args.weights): 
    print("Loading weights")
    st_model.load_weights(args.weights)

print('Compiling model')
adam = Adam(lr=args.lr, clipnorm=5.) # Clipping the norm avoid gradient explosion, no needs to suffocate it
st_model.compile(adam, loss='mse') # loss=frobenius_error (this is not giving the same loss)

print('Training model')
history = st_model.fit(X, y, batch_size=args.batch_size, nb_epoch=args.nb_epoch, verbose=1, validation_data=(X_cv, y_cv))
losses = history.history

if args.nb_epoch > 0:
    print("Saving final data")
    prefixedDir = prefixed_dir = "%s/%s-%s-%s" % (results_dir, str(int(time.time())), K._BACKEND, dim_ordering)
    export_model(st_model, prefixedDir)
    with open(prefixedDir + '/losses.json', 'w') as outfile:
        json.dump(losses, outfile)  
    plot_losses(losses, prefixedDir)