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
        export_model(model, data_model_folder)

        os.remove(data_model_folder + '/archi.json')
        os.remove(data_model_folder + '/last_weights.hdf5')
        os.rmdir(data_model_folder)

    def test_import_model(self):
        data_model_folder = dir + "/../fixture/model_import"
        model = import_model(data_model_folder)

        input_img = np.array([load_image_st(dir + '/../fixture/blue.png', size=None, verbose=False)])
        output = model.predict([input_img]).astype('int8')
        true_output = np.array([
            [
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ],
                [
                    [  62,   -2,  103],
                    [ 108,   25, -103],
                    [-103,  116, -125]
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 1, 52]
                ]
            ]
        ])

        self.assertEqual(len(model.layers),3)
        self.assertEqual(True, (output==true_output).all())

if __name__ == '__main__':
    unittest.main()