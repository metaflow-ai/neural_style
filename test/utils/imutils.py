import os
from utils.imutils import *
import unittest

dir = os.path.dirname(os.path.realpath(__file__))

class TestImUtils(unittest.TestCase):

    def test_load_images(self):
        files = load_images(dir + '/../fixture')
        self.assertEqual(files.shape, (4, 3, 256, 256))

    def test_load_images_limit(self):
        file = load_images(dir + '/../fixture', 1)
        self.assertEqual(file.shape, (1, 3, 256, 256))

    def test_create_noise_tensor(self):
        file = create_noise_tensor(3, 4, 5)
        self.assertEqual(file.shape, (1, 3, 4, 5))
        

if __name__ == '__main__':
    unittest.main()