import os, sys, unittest
 
dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

from keras import backend as K
from scipy import misc

from utils.lossutils import *


dir = os.path.dirname(os.path.realpath(__file__))

class TestLossUtils(unittest.TestCase):

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
        grad = K.gradients(loss, x)
        get_grads = K.function([x], grad)

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

        self.assertEqual(True, (get_grads([input])==true_grad).all())

    def test_grams_th(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('th')

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
        get_grams = K.function([x], [gram_mat])        
        K.set_image_dim_ordering(previous_image_dim_ordering)

        pred_grams = get_grams([input])[0]
        self.assertEqual(True, (pred_grams==true_grams).all())

    def test_grams_tf(self):
        previous_image_dim_ordering = K.image_dim_ordering()
        K.set_image_dim_ordering('tf')

        input = np.zeros((1, 3, 4, 4))
        iter = 0
        for i in range(input.shape[1]):
            for j in range(input.shape[2]):
                for k in range(input.shape[3]):
                    input[0][i][j][k] = iter
                    iter += 1
        input = input.astype(K.floatx())
        input = np.transpose(input, (0, 2, 3, 1))

        true_grams = np.array([[
                    [1240, 3160, 5080],
                    [3160, 9176,15192],
                    [5080,  15192,  25304]
                ]]).astype(K.floatx())
        true_grams /= input.shape[1] * input.shape[2] * input.shape[3]

        x = K.placeholder(input.shape, name='x')
        gram_mat = grams(x)
        get_grams = K.function([x], [gram_mat])
        K.set_image_dim_ordering(previous_image_dim_ordering)   

        pred_grams = get_grams([input])[0]
        self.assertEqual(True, (pred_grams==true_grams).all())

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
        get_loss = K.function([x], [loss])
        
        error = get_loss([input])[0]
        true_error = 60344.299382716
        
        self.assertEqual(np.round(error.item(0)), np.round(true_error))

if __name__ == '__main__':
    unittest.main()