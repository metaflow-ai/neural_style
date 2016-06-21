import os, sys, unittest
from scipy import misc
import numpy as np

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

from keras import backend as K
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Input
from keras.models import Model

from utils.general import export_model, import_model
from utils.imutils import load_image_st

dir = os.path.dirname(os.path.realpath(__file__))

class TestImUtils(unittest.TestCase):

    def test_export_model(self):
        input = Input(shape=(3, 4, 4), name='input', dtype='float32')
        out = Convolution2D(3, 3, 3, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
        out = Activation('relu')(out)
        model = Model(input=[input], output=[out])

        data_model_folder = dir + "/../fixture/model_export"
        if K._BACKEND == 'tensorflow':
            import tensorflow as tf
            saver = tf.train.Saver()
        else:
            saver = None
        export_model(model, data_model_folder, saver=saver)

        os.remove(data_model_folder + '/archi.json')
        os.remove(data_model_folder + '/last_weights.hdf5')
        if K._BACKEND == 'tensorflow':
            os.remove(data_model_folder + '/checkpoint')
            os.remove(data_model_folder + '/tf-last_weights')
            os.remove(data_model_folder + '/tf-last_weights.meta')
            os.remove(data_model_folder + '/tf-model_graph')
            os.remove(data_model_folder + '/tf-frozen_model.pb')
        os.rmdir(data_model_folder)

    def test_import_model(self):
        data_model_folder = dir + "/../fixture/model_conv2d_relu"

        should_convert = K._BACKEND == "theano"
        model = import_model(data_model_folder, should_convert=should_convert)
        input_img = np.array([load_image_st(dir + '/../fixture/blue.png', size=None, verbose=False)])

        

        output = model.predict([input_img]).astype('int32')
        true_output = np.array([
            [
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [
                    [131, 116, 153],
                    [153, 281, 364],
                    [103, 254, 318]
                ],
                [
                    [52, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ]
            ]
        ])

        self.assertEqual(len(model.layers),3)
        self.assertEqual(True, (output==true_output).all())

if __name__ == '__main__':
    unittest.main()