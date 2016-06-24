import os, sys, unittest
import numpy as np

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

from utils.imutils import (load_images, load_image, load_mean, 
                            preprocess, deprocess, create_noise_tensor, resize)
from scipy import misc

from keras import backend as K

dir = os.path.dirname(os.path.realpath(__file__))

class TestImUtils(unittest.TestCase):

    def test_load_mean_tf(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('tf')
        mean = load_mean()
        real_mean = np.array([[[[103.939, 116.779, 123.68]]]])
        K.set_image_dim_ordering(previous_image_dim_ordering)

        self.assertEqual(True, (mean==real_mean).all())

    def test_load_mean_th(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('th')
        mean = load_mean()
        real_mean = np.array([[[[103.939]], [[116.779]], [[123.68]]]])
        K.set_image_dim_ordering(previous_image_dim_ordering)

        self.assertEqual(True, (mean==real_mean).all())

    # def test_load_mean_exception(self):
    #     self.assertRaises(Exception, load_mean('test'))

    def test_preprocess_tf(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('tf')
        blue_im = misc.imread(dir + '/../fixture/blue.png')
        red_im = np.array(misc.imread(dir + '/../fixture/red.png').astype(K.floatx()))
        red_im = (red_im - load_mean()[0]).astype('uint8')
        new_red_im = preprocess(blue_im).astype('uint8')
        K.set_image_dim_ordering(previous_image_dim_ordering)

        self.assertEqual(True, (red_im==new_red_im).all())

    def test_load_image(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('tf')
        blue_im = load_image(dir + '/../fixture/blue.png')
        K.set_image_dim_ordering(previous_image_dim_ordering)

        self.assertEqual(blue_im.shape, (600, 600, 3))

    def test_load_image_th(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('th')
        blue_im = load_image(dir + '/../fixture/blue.png')
        K.set_image_dim_ordering(previous_image_dim_ordering)

        self.assertEqual(blue_im.shape, (3, 600, 600))

    def test_load_images(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('tf')
        files = load_images(dir + '/../fixture')
        K.set_image_dim_ordering(previous_image_dim_ordering)

        self.assertEqual(files.shape, (4, 600, 600, 3))

    def test_load_images_limit(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('tf')
        file = load_images(dir + '/../fixture', 1)
        K.set_image_dim_ordering(previous_image_dim_ordering)

        self.assertEqual(file.shape, (1, 600, 600, 3))

    def test_deprocess(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('tf')
        blue_im = misc.imread(dir + '/../fixture/blue.png')
        im = preprocess(blue_im)
        im = deprocess(im)
        K.set_image_dim_ordering(previous_image_dim_ordering)

        self.assertEqual(True, (blue_im==im).all())

    def test_deprocess_th(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('th')
        blue_im = misc.imread(dir + '/../fixture/blue.png')
        im = preprocess(blue_im)
        im = deprocess(im)
        K.set_image_dim_ordering(previous_image_dim_ordering)

        self.assertEqual(True, (blue_im==im).all())

    def test_create_noise_tensor(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('tf')
        file = create_noise_tensor(4, 5 ,3)
        K.set_image_dim_ordering(previous_image_dim_ordering)

        self.assertEqual(file.shape, (1, 4, 5, 3))

    def test_resize(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('tf')
        
        ims = load_images(dir + '/../fixture')
        ims = resize(ims, (150, 150))

        K.set_image_dim_ordering(previous_image_dim_ordering)

        self.assertEqual(ims.shape, (4, 150, 150, 3))

        

if __name__ == '__main__':
    unittest.main()