import random
import h5py, os, sys
import numpy as np

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

from keras import backend as K
if K._BACKEND == 'tensorflow':
    import tensorflow as tf
    import freeze_graph as freeze_tools
    
from keras.models import model_from_json
from keras.utils.np_utils import convert_kernel
from vgg19.model_headless import get_layer_data

from imutils import load_image, get_image_list, preprocess

dir = os.path.dirname(os.path.realpath(__file__))

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
        if k + offset >= nbLayer or (limit > 0 and k >= limit):
            break
        g = f['layer_{}'.format(k)]
        weights = [g.get('param_{}'.format(p))[()] for p in range(g.attrs['nb_params'])]
        print(model.layers[k+offset].name)
        if len(weights):
            if K._BACKEND == 'theano': 
                # I have to do it for theano because theano flips the convolutional kernel
                # and i don't have access to the API of the conv ops
                weights[0] = np.round(weights[0][:, :, ::-1, ::-1], 4)
            if K.image_dim_ordering() == 'tf':
                weights[0] = np.transpose(weights[0], [2, 3, 1, 0])
            # print(weights[0].shape)
        model.layers[k+offset].set_weights(weights)
    f.close()

    model.save_weights(outputFilename, overwrite=True)

def import_model(absolute_model_dir, best=True, should_convert=False, custom_objects={}):
    archi_json = open(absolute_model_dir + '/archi.json').read()
    model = model_from_json(archi_json, custom_objects)

    if os.path.isfile(absolute_model_dir + '/best_weights.hdf5') and best:
        model.load_weights(absolute_model_dir + '/best_weights.hdf5')
    else:
        model.load_weights(absolute_model_dir + '/last_weights.hdf5')

    if should_convert == True:
        if K._BACKEND == 'tensorflow':
            ops = []
            for layer in model.layers:
               if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:
                  original_w = K.get_value(layer.W)
                  converted_w = convert_kernel(original_w)
                  ops.append(tf.assign(layer.W, converted_w).op)
            K.get_session().run(ops)
        else:
            for layer in model.layers:
               if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:
                  original_w = K.get_value(layer.W)
                  converted_w = convert_kernel(original_w)
                  K.set_value(layer.W, converted_w)


    return model
    
def export_model(model, absolute_model_dir, best_weights=None, saver=None, global_step=None):
    if not os.path.isdir(absolute_model_dir): 
        os.makedirs(absolute_model_dir)

    model.save_weights(absolute_model_dir + "/last_weights.hdf5", overwrite=True)
    if K._BACKEND == 'tensorflow' and saver != None:
        sess = K.get_session()
        saver.save(sess, absolute_model_dir + '/tf-last_weights', global_step=global_step)

    if best_weights != None:
        model.set_weights(best_weights)
        model.save_weights(absolute_model_dir + "/best_weights.hdf5", overwrite=True)
        if K._BACKEND == 'tensorflow' and saver != None:
            saver.save(sess, absolute_model_dir + '/tf-best_weights', global_step=global_step)

    # Graph
    json = model.to_json()
    open(absolute_model_dir + "/archi.json", 'w').write(json)
    if K._BACKEND == 'tensorflow' and saver != None and global_step == None:
        graph_def = sess.graph.as_graph_def()
        tf.train.write_graph(graph_def, absolute_model_dir, 'tf-model_graph')

        freeze_graph(model, absolute_model_dir, best_weights)

def freeze_graph(model, absolute_model_dir, best_weights=None):
    input_graph_path = absolute_model_dir + '/tf-model_graph'
    input_saver_def_path = ""
    input_binary = False
    if best_weights != None:
        input_checkpoint_path = absolute_model_dir + '/tf-best_weights'
    else:
        input_checkpoint_path = absolute_model_dir + '/tf-last_weights'
    output_node_names = model.output.name.split(':')[0]
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = absolute_model_dir + "/tf-frozen_model.pb"
    clear_devices = True

    freeze_tools.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_path,
                              clear_devices, "", verbose=False)

def get_shape(x):
    if isinstance(x, (np.ndarray)):
        return x.shape 
    elif K._BACKEND == 'theano':
        return K.shape(x)
    else:
        try:
            return K.int_shape(x)
        except Exception:
            return K.shape(x)

def mask_data(data, selector):
    return [d for d, s in zip(data, selector) if s]

def generate_data_from_image_list(image_list, size, style_fullpath_prefix, input_len=1, output_len=1, batch_size=4, transform_f=None, preprocess_type='none', verbose=False):
    if transform_f != None:
        file = h5py.File(style_fullpath_prefix + '_' + str(size[0]) + '.hdf5', 'r')
        y_style1 = np.array(file.get('conv_1_2'))
        y_style2 = np.array(file.get('conv_2_2'))
        y_style3 = np.array(file.get('conv_3_4'))
        y_style4 = np.array(file.get('conv_4_2'))

    # Init inputs/outputs
    inputs = []
    outputs = []
    for i in range(input_len):
        inputs.append([])
    for i in range(output_len):
        outputs.append([])
    nb_element = 0
    while 1:
        random.shuffle(image_list)
        for fullpath in image_list:
            nb_element += 1
            
            im = load_image(fullpath, size, preprocess_type=preprocess_type, verbose=verbose)

            if transform_f != None:
                f_input = [ preprocess(np.array([im]), type='vgg19') ]
                y_content = transform_f(f_input)[0][0] # First element in the list of result, first element of the output batch
                inputs[0].append(im)
                outputs[0].append(y_content)
                outputs[1].append(y_style1)
                outputs[2].append(y_style2)
                outputs[3].append(y_style3)
                outputs[4].append(y_style4)
                outputs[5].append(np.zeros_like(im))
            else:
                for i in range(input_len):
                    inputs[i].append(im)
                for i in range(output_len):
                    outputs[i].append(im)

            if nb_element >= batch_size:
                nb_element = 0

                inputs_list = []
                for i in range(input_len):
                    inputs_list.append(np.array(inputs[i]))
                outputs_list = []
                for i in range(output_len):
                    outputs_list.append(np.array(outputs[i]))

                yield(inputs_list, outputs_list)

                # reset inputs/outputs
                for i in range(input_len):
                    inputs[i] = []
                for i in range(output_len):
                    outputs[i] = []

            
            
