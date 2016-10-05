from keras import backend as K
from keras.engine import Layer

if K._BACKEND == 'theano':
    raise Exception('theano backend is not supported')
else:
    import tensorflow as tf

class InstanceNormalization(Layer):
    def __init__(self, name, **kwargs):
        self.supports_masking = False
        self.name = name

        super(InstanceNormalization, self).__init__(**kwargs)


    def call(self, X, mask=None):
        (mean, variance) = tf.nn.moments(X, [1, 2], keep_dims=True)
        shape = X.get_shape().as_list()
        # shape[0] = tf.shape(X)[0]
        # offset = tf.get_variable('%s-offset' % self.name, shape=[shape[0], 1, 1, shape[3]])
        # scale = tf.get_variable('%s-scale' % self.name, shape=[shape[0], 1, 1, shape[3]])
        variance_epsilon = 1e-7

        A = tf.nn.batch_normalization(X, mean, variance, None, None, variance_epsilon)
        Y = tf.nn.relu(A)

        return Y

    def get_config(self):
        config = {
            'name': self.name
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))