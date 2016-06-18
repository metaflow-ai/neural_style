import h5py, os
import numpy as np

from keras import backend as K
if K._BACKEND == 'tensorflow':
    import tensorflow as tf
from keras.models import model_from_json

from imutils import load_image, load_image_st, get_image_list

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
            if K._BACKEND == 'theano': 
                # I have to do it for theano because theano flips the convolutional kernel
                # and i don't have access to the API of the conv ops
                weights[0] = np.round(weights[0][:, :, ::-1, ::-1], 4)
            # print(weights[0].shape)
        model.layers[k+offset].set_weights(weights)
    f.close()

    model.save_weights(outputFilename, overwrite=True)

def export_model(model, absolute_model_dir, best_weights=None):
    if not os.path.isdir(absolute_model_dir): 
        os.makedirs(absolute_model_dir)

    model.save_weights(absolute_model_dir + "/last_weights.hdf5", overwrite=True)
    if K._BACKEND == 'tensorflow':
        sess = K.get_session()
        saver = tf.train.Saver(var_list=None)
        saver.save(sess, absolute_model_dir + '/tf-last_weights', global_step=None)

    if best_weights != None:
        model.set_weights(best_weights)
        model.save_weights(absolute_model_dir + "/best_weights.hdf5", overwrite=True)
        if K._BACKEND == 'tensorflow':
            saver = tf.train.Saver(var_list=None)
            saver.save(sess, absolute_model_dir + '/tf-best_weights', global_step=None)

    # Graph
    open(absolute_model_dir + "/archi.json", 'w').write(model.to_json())
    if K._BACKEND == 'tensorflow':
        graph_def = sess.graph.as_graph_def()
        tf.train.write_graph(graph_def, absolute_model_dir, 'tf-model_graph')


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

def generate_data_from_image_list(image_folder, size, style_fullpath_pefix, dim_ordering='tf', verbose=False, st=False):
    image_list = get_image_list(image_folder)
    file = h5py.File(style_fullpath_pefix + '_' + str(size[0]) + '.hdf5', 'r')
    y_style1 = np.array(file.get('conv_1_2'))
    y_style2 = np.array(file.get('conv_2_2'))
    y_style3 = np.array(file.get('conv_3_4'))
    y_style4 = np.array(file.get('conv_4_2'))
    while 1:
        for fullpath in image_list:
            if st:
                im = load_image_st(fullpath, size, verbose)
            else:
                im = load_image(fullpath, size, dim_ordering, verbose)
            hdf5_filepath = fullpath.split('/')
            filename = hdf5_filepath.pop().split('.')[0]
            hdf5_filepath = '/'.join(hdf5_filepath) + '/results/' + filename + '_' + str(size[0]) + '.hdf5'
            y_content = np.array(h5py.File(hdf5_filepath, 'r').get('conv_3_2'))
            
            yield ([im], [y_content, y_style1, y_style2, y_style3, y_style4, np.zeros(im)])