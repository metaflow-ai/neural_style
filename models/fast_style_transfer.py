import os, sys, argparse

import tensorflow as tf
import numpy as np
from tensorflow-vgg import vgg19

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

def instance_norm_layer(name, X):
    with tf.variable_scope(name):
        (mean, variance) = tf.nn.moments(X, [1, 2], keep_dims=True)
        shape = tf.shape(X)
        offset = tf.get_variable('%s-offset' % name, shape=[shape[0], 1, 1, shape[3]])
        scale = tf.get_variable('%s-scale' % name, shape=[shape[0], 1, 1, shape[3]])
        variance_epsilon = 1e-7

        A = tf.nn.batch_normalization(X, mean, variance, offset, scale, variance_epsilon)
        Y = tf.nn.relu(A)

    return Y

def res_layer(name, X):
    batch_dim, height, width, channels = tf.shape(X)
    with tf.variable_scope(name):
        W1 =  tf.get_variable('%s-W1' % name, shape=[3, 3, channels, channels])
        b1 = tf.get_variable('%s-b1', shape=[channels])
        A1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME") + b1
        Z1 = tf.nn.relu(A1)
        N1 = instance_norm_layer(Z1)

        W2 =  tf.get_variable('%s-W2' % name, shape=[3, 3, channels, channels])
        b2 = tf.get_variable('%s-b2', shape=[channels])
        A2 = tf.nn.conv2d(N1, W2, strides=[1, 1, 1, 1], padding="SAME") + b2
        Z2 = tf.nn.relu(A2)
        N2 = instance_norm_layer(Z2)

        Y = tf.add(X, N2)

    return Y

def subpixel_layer(name, X, r, color=False):
    def _phase_shift(I, r):
        bsize, a, b, c = I.shape().as_list()
        bsize = tf.shape(I)[0] # I've made a minor change to the original implementation to enable Dimension(None) for the batch dim
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(1, a, X)  # a, [bsize, b, r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
        X = tf.split(1, b, X)  # b, [bsize, a*r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
        return tf.reshape(X, (bsize, a*r, b*r, 1))

    with tf.variable_scope(name):
        if color:
            Xc = tf.split(3, 3, X)
            Y = tf.concat(3, [_phase_shift(x, r) for x in Xc])
        else:
            Y = _phase_shift(X, r)
    return Y


# inputs th ordering, BGR
def fast_style_transfer(input_shape, ratio=4, nb_res_layer=5):
    with tf.variable_scope('Placeholder'):
        X = tf.placeholder(tf.int32, shape=input_shape, name='Inputs_placeholder')

    with tf.variable_scope('Encoder'):
        W1 = tf.get_variable('encoder-W1', shape=[3, 3, 3, 32])
        b1 = tf.get_variable('encoder-b1', shape=[32])
        A1 = tf.nn.conv2d(X, W1, strides=[1, 2, 2, 1], padding="SAME") + b1
        Z1 = tf.nn.relu(A1)
        Y = instance_norm_layer(Z1)

        W2 = tf.get_variable('encoder-W2', shape=[3, 3, 32, 128])
        b2 = tf.get_variable('encoder-b2', shape=[128])
        A2 = tf.nn.conv2d(Y, W2, strides=[1, 2, 2, 1], padding="SAME") + b2
        Z2 = tf.nn.relu(A2)
        Y = instance_norm_layer(Z2)

        if ratio is 8:
            W3 = tf.get_variable('encoder-W3', shape=[3, 3, 128, 256])
            b3 = tf.get_variable('encoder-b3', shape=[256])
            A3 = tf.nn.conv2d(Y, W3, strides=[1, 2, 2, 1], padding="SAME") + b3
            Z3 = tf.nn.relu(A3)
            Y = instance_norm_layer(Z3)

    for i in range(nb_res_layer):
        Y = res_layer('residual_layer_%d' % i, Y)

    with tf.variable_scope('Decoder'):
        if ratio is 8:
            W_decoder = tf.get_variable('encoder-W', shape=[3, 3, 256, 192])
            b_decoder = tf.get_variable('encoder-b', shape=[192])
        else:
            W_decoder = tf.get_variable('encoder-W', shape=[3, 3, 256, 48])
            b_decoder = tf.get_variable('encoder-b', shape=[48])

        Z = tf.nn.conv2d(Y, W_decoder, strides=[1, 1, 1, 1], padding="SAME") + b_decoder
        Y = subpixel_layer('subpixel_layer', Z, ratio, color=True)

    return {
        'input': X,
        'output': Y
    }


if __name__ == "__main__":
    data_dir = dir + '/../data'

    parser = argparse.ArgumentParser(
        description='Neural artistic style. Generates an image by combining '
                    'the content of an image and the style of another.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--content', default=data_dir + '/overfit/COCO_val2014_000000000074.jpg', type=str, help='Content image.')
    parser.add_argument('--style', default=data_dir + '/paintings/edvard_munch-the_scream.jpg', type=str, help='Style image.')
    parser.add_argument('--image_size', default=256, type=int, help='Input image size.')
    parser.add_argument('--ratio', default=4, type=int, help='Ratio between encoding and decoding')
    parser.add_argument('--nb_res_layer', default=5, type=int, help='Number of residual layer.')
    args = parser.parse_args()

    dir = os.path.dirname(os.path.realpath(__file__))
    results_dir = dir + '/data/st'

    input_shape = [None, args.image_size, args.image_size, 3]
    fst = fast_style_transfer(input_shape, ratio=args.ratio, nb_res_layer=args.nb_res_layer)
    tf.image_summary('input_img', fst['input'], 2)
    tf.image_summary('output_img', fst['output'], 2)

    vgg = vgg19.Vgg19()


    style_loss = 1
    content_loss = 1
    tv_loss = 1
    tf.scalar_summary('style_loss', style_loss)
    tf.scalar_summary('content_loss', content_loss)
    tf.scalar_summary('tv_loss', tv_loss)
    
    adam = tf.AdamOptimizer(1e-3)
    train_op = adam.minimize(total_loss)

    for i in range(10):
        style_coef = np.random.random_integers(50,150)
        content_coef = np.random.random_integers(5,10)
        tv_coef = np.random.random_integers(5,10)

        total_loss = style_coef * style_loss + content_coef * content_loss + tv_coef * tv_loss
        
        tf.scalar_summary('total_loss', total_loss)

        summaries_op = tf.merge_all_summaries()
