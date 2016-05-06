import h5py

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
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        print(model.layers[k+offset].name)
        model.layers[k+offset].set_weights(weights)

    f.close()

    model.save_weights(outputFilename)