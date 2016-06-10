import h5py, os
import numpy as np

from keras.models import model_from_json

# You don't need any input layer in a sequential model which usually end up 
# with a model minus that one input layer
def copySeqWeights(model, weightsFullPath, outputFilename, offset=1, limit=-1):
    nbLayer = len(model.layers)
    f = h5py.File(weightsFullPath, "r")
    # print(f.name, f.keys())
    # print(f['layer_1'].keys(), f['layer_1']['param_0'])

    nbLayers = f.attrs['nb_layers']
    print("Number of layers in the original weight: ", nbLayers)
    print("Number of layers in the model: ", len(model.layers))
    for k in range(nbLayers):
        if k >= nbLayer - offset or (limit > 0 and k >= limit):
            break
        g = f['layer_{}'.format(k)]
        weights = [g.get('param_{}'.format(p))[()] for p in range(g.attrs['nb_params'])]
        print(model.layers[k+offset].name)
        if len(weights):
            weights[0] = np.round(weights[0][:, :, ::-1, ::-1], 4)
            # print(weights[0].shape)
        model.layers[k+offset].set_weights(weights)
    f.close()

    model.save_weights(outputFilename, overwrite=True)

def export_model(model, absolute_model_dir, best_weights=None):
    if not os.path.isdir(absolute_model_dir): 
        os.makedirs(absolute_model_dir)

    model.save_weights(absolute_model_dir + "/last_weights.hdf5", overwrite=True)
    if best_weights != None:
        model.set_weights(best_weights)
        model.save_weights(absolute_model_dir + "/best_weights.hdf5", overwrite=True)
    open(absolute_model_dir + "/archi.json", 'w').write(model.to_json())

def import_model(absolute_model_dir, best=True, custom_objects={}):
    archi_json = open(absolute_model_dir + '/archi.json').read()
    model = model_from_json(archi_json, custom_objects)

    if os.path.isfile(absolute_model_dir + '/best_weights.hdf5') and best:
        model.load_weights(absolute_model_dir + '/best_weights.hdf5')
    else:
        model.load_weights(absolute_model_dir + '/last_weights.hdf5')

    return model

def mask_data(data, selector):
    return [d for d, s in zip(data, selector) if s]