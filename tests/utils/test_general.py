import os, sys, unittest
from scipy import misc

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

from keras import backend as K
from keras.layers.convolutional import (Convolution2D, UpSampling2D)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Input
from keras.models import Model

from utils.general import export_model, import_model

dir = os.path.dirname(os.path.realpath(__file__))
data_model_folder = dir + "/../fixture/model"

class TestImUtils(unittest.TestCase):

    def test_export_model(self):

        input = Input(shape=(3, 4, 4), name='input', dtype='float32')
        out = Convolution2D(32, 3, 3, 
            init='he_normal', subsample=(1, 1), border_mode='same', activation='linear')(input)
        out = Activation('relu')(out)
        model = Model(input=[input], output=[out])

        export_model(model, data_model_folder)

    def test_import_model(self):

        model = import_model(data_model_folder)
        self.assertEqual(len(model.layers),3)


if __name__ == '__main__':
    unittest.main()