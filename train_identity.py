import os, json, time, argparse

from keras import backend as K
from keras.engine.training import collect_trainable_weights
from keras.optimizers import Adam
# from keras.utils.visualize_util import plot as plot_model

from models.style_transfer import (style_transfer_conv_transpose)

from utils.imutils import plot_losses
from utils.lossutils import (frobenius_error, train_weights)
from utils.general import export_model

dir = os.path.dirname(os.path.realpath(__file__))
dataDir = dir + '/data'
resultsDir = dir + '/models/data/identity'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)
trainDir = dataDir + '/train'
overfitDir = dataDir + '/overfit'

parser = argparse.ArgumentParser(
    description='Neural artistic style. Generates an image by combining '
                'the content of an image and the style of another.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--batch_size', default=8, type=int, help='batch size.')
parser.add_argument('--image_size', default=256, type=int, help='Input image size.')
parser.add_argument('--max_iter', default=2000, type=int, help='Number of training iter.')
parser.add_argument('--nb_res_layer', default=6, type=int, help='Number of residual layers in the style transfer model.')
args = parser.parse_args()

channels = 3
width = args.image_size
height = args.image_size
input_shape = (channels, width, height)
batch_size = args.batch_size

identityWeightsFullpath = dir + '/models/identity_weights.hdf5'
st_model = style_transfer_conv_transpose(input_shape=input_shape, nb_res_layer=args.nb_res_layer) # th ordering, BGR
if os.path.isfile(identityWeightsFullpath): 
    print("Loading weights")
    st_model.load_weights(identityWeightsFullpath)

train_loss = frobenius_error(st_model.input, st_model.output)

print('Compiling Adam update')
adam = Adam(lr=5e-02)
updates = adam.get_updates(collect_trainable_weights(st_model), st_model.constraints, train_loss)

print('Compiling train function')
inputs = [st_model.input, K.learning_phase()]
outputs = [train_loss]
train_iteratee = K.function(inputs, outputs, updates=updates)

print('Starting training')
weights, losses = train_weights(
    # trainDir,
    overfitDir,
    (height, width),
    st_model, 
    train_iteratee, 
    max_iter=args.max_iter,
    batch_size=batch_size
)
best_weights = weights[0]
last_weights = weights[1]

print("Saving final data")
prefixedDir = resultsDir + '/' + str(int(time.time()))
export_model(st_model, prefixedDir, best_weights)
with open(prefixedDir + '/losses.json', 'w') as outfile:
    json.dump(losses, outfile)  
plot_losses(losses, prefixedDir)
