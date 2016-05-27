import os, sys, unittest
import theano 
dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

from keras import backend as K
from scipy import misc

from utils.lossutils import *


dir = os.path.dirname(os.path.realpath(__file__))

class TestImUtils(unittest.TestCase):

    def test_total_variation_error(self):
        # Prepare input
        input = np.zeros((1, 3, 4, 4))
        iter = 0
        for i in range(input.shape[1]):
            for j in range(input.shape[2]):
                for k in range(input.shape[3]):
                    input[0][i][j][k] = iter
                    iter += 1
        input = input.astype('float32')

        x = K.placeholder(input.shape, name='x')
        loss = total_variation_error(x, 2)
        gLoss = K.gradients(loss, x)
        get_grads = theano.function([x], gLoss)

        # GradInput result for beta = 2
        true_grad = np.array([[
            [
                [-5, -4, -4,  1],
                [-1, 0, 0, 1],
                [-1, 0, 0, 1],
                [4, 4, 4, 0]
            ],
            [
                [-5, -4, -4,  1],
                [-1, 0, 0, 1],
                [-1, 0, 0, 1],
                [4, 4, 4, 0]
            ],
            [
                [-5, -4, -4,  1],
                [-1, 0, 0, 1],
                [-1, 0, 0, 1],
                [4, 4, 4, 0]
            ],
        ]]).astype(K.floatx())

        self.assertEqual(True, (get_grads(input)==true_grad).all())

    def test_grams(self):
        input = np.zeros((1, 3, 4, 4))
        iter = 0
        for i in range(input.shape[1]):
            for j in range(input.shape[2]):
                for k in range(input.shape[3]):
                    input[0][i][j][k] = iter
                    iter += 1
        input = input.astype(K.floatx())

        true_grams = np.array([[
                    [1240, 3160, 5080],
                    [3160, 9176,15192],
                    [5080,  15192,  25304]
                ]]).astype(K.floatx())
        true_grams /= input.shape[1] * input.shape[2] * input.shape[3]

        x = K.placeholder(input.shape, name='x')
        gram_mat = grams(x)
        get_grams = theano.function([x], gram_mat)        

        self.assertEqual(True, (get_grams(input)==true_grams).all())

    def test_grams_loss(self):
        input = np.zeros((1, 3, 4, 4))
        iter = 0
        for i in range(input.shape[1]):
            for j in range(input.shape[2]):
                for k in range(input.shape[3]):
                    input[0][i][j][k] = iter
                    iter += 1
        input = input.astype(K.floatx())


        x = K.placeholder(input.shape, name='x')
        gram_mat = grams(x)
        loss = frobenius_error(gram_mat, np.ones((1, 3, 3)))
        get_loss = theano.function([x], loss)
        
        error = get_loss(input)
        true_error = 60344.299382716
        
        self.assertEqual(np.round(error.item(0)), np.round(true_error))

if __name__ == '__main__':
    unittest.main()