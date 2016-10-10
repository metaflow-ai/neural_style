from keras import backend as K
from keras.engine import Layer

if K._BACKEND == 'theano':
    raise Exception('theano backend is not supported')
else:
    import tensorflow as tf

class PhaseShift(Layer):
    def __init__(self, ratio=4, color=False, **kwargs):
        self.supports_masking = False
        self.ratio = ratio
        self.color = color

        super(PhaseShift, self).__init__(**kwargs)

    def _phase_shift(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.shape(I)[0]
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(1, a, X)  # a, [bsize, b, r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
        X = tf.split(1, b, X)  # b, [bsize, a*r, r]
        X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, a*r, b*r
        return tf.reshape(X, (bsize, a*r, b*r, 1))

    def call(self, X, mask=None):
        total_channels = X.get_shape()[3]
        if total_channels % (self.ratio * self.ratio) != 0:
            raise ValueError('total_channels % (self.ratio * self.ratio) must equal to 0')

        nb_output_channels = total_channels // (self.ratio * self.ratio)
        Xc = tf.split(3, nb_output_channels, X)
        Y = tf.concat(3, [self._phase_shift(x, self.ratio) for x in Xc])

        return Y

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1] * self.ratio, input_shape[2] * self.ratio, input_shape[3] // (self.ratio * self.ratio))

    def get_config(self):
        config = {
            'ratio': self.ratio,
            'color': self.color,
        }
        base_config = super(PhaseShift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))