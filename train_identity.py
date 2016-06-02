import os, json

from keras import backend as K
from keras.engine.training import collect_trainable_weights
from keras.optimizers import Adam
# from keras.utils.visualize_util import plot as plot_model

from models.style_transfer import (style_transfer_conv_transpose)

from utils.imutils import plot_losses
from utils.lossutils import (frobenius_error, train_weights)

dir = os.path.dirname(os.path.realpath(__file__))
dataDir = dir + '/data'
resultsDir = dir + '/models/weights'
if not os.path.isdir(resultsDir): 
    os.makedirs(resultsDir)
trainDir = dataDir + '/train'
overfitDir = dataDir + '/overfit'

channels = 3
width = 256
height = 256
input_shape = (channels, width, height)
batch_size = 8
identityWeightsFullpath = dir + '/models/identity_weights.hdf5'
st_model = style_transfer_conv_transpose(input_shape=input_shape) # th ordering, BGR
if os.path.isfile(identityWeightsFullpath): 
    print("Loading weights")
    st_model.load_weights(identityWeightsFullpath)

train_loss = frobenius_error(st_model.input, st_model.output)

print('Iterating over hyper parameters')
current_iter = 0
for alpha in [20e0]:
    for beta in [1.]:
        for gamma in [1e-04]:
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
                cv_input_dir=None, 
                max_iter=500,
                batch_size=batch_size
            )

            best_trainable_weights = weights[0]
            last_trainable_weights = weights[1]
            prefix = str(current_iter).zfill(4)
            archi = resultsDir + '/' + prefix + 'archi.json'
            best_st_weights = resultsDir + '/' + prefix + 'best_identity_weights.hdf5'
            last_st_weights = resultsDir + '/' + prefix + 'last_identity_weights.hdf5'
            fullpath_loss = resultsDir + '/' + prefix + 'identity_loss.json'
            current_iter += 1

            print("Saving final data")
            st_model.set_weights(best_trainable_weights)
            st_model.save_weights(best_st_weights, overwrite=True)
            st_model.set_weights(last_trainable_weights)
            st_model.save_weights(last_st_weights, overwrite=True)
            open(archi, 'w').write(st_model.to_json())

            with open(fullpath_loss, 'w') as outfile:
                json.dump(losses, outfile)  

            plot_losses(losses, resultsDir, prefix)
