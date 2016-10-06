from keras import backend as K
from keras.engine import Layer, InputSpec

if K._BACKEND == 'theano':
    raise Exception('theano backend is not supported')
else:
    import tensorflow as tf


class ReflectPadding2D(Layer):
    def __init__(self, padding=(1, 1), dim_ordering=K.image_dim_ordering(), **kwargs):
        super(ReflectPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            width = input_shape[2] + 2 * self.padding[0] if input_shape[2] is not None else None
            height = input_shape[3] + 2 * self.padding[1] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.dim_ordering == 'tf':
            width = input_shape[1] + 2 * self.padding[0] if input_shape[1] is not None else None
            height = input_shape[2] + 2 * self.padding[1] if input_shape[2] is not None else None
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        return tf.pad(x, [
            [ 0, 0 ], 
            [ self.padding[0], self.padding[0] ], 
            [ self.padding[1], self.padding[1] ], 
            [ 0, 0 ]
        ], "REFLECT")

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))