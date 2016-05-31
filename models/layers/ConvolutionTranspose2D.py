from keras import backend as K
from keras.backend.common import _FLOATX
from keras import activations, initializations, regularizers, constraints
from keras.engine import Layer, InputSpec

if K._BACKEND == 'theano':
    import theano
    import theano.tensor as T
else:
    import tensorflow as tf

def conv_transpose_out_length(input_length, filter_size, border_mode, stride):
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    if border_mode == 'valid':
        output_length = (input_length - 1) * stride + filter_size
    elif border_mode == 'same':
        output_length = input_length * stride
    return output_length

class ConvolutionTranspose2D(Layer):
    '''Convolution transpose operator for filtering windows of two-dimensional inputs.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

    # Examples

    ```python
        # apply a 3x3 convolution transpose with 64 output filters on a 256x256 image:
        model = Sequential()
        model.add(ConvolutionTranspose2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
        # now model.output_shape == (None, 64, 256, 256)

        # add a 3x3 convolution on top, with 32 output filters:
        model.add(ConvolutionTranspose(32, 3, 3, border_mode='same'))
        # now model.output_shape == (None, 32, 256, 256)
    ```

    # Arguments
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid' or 'same'.
        subsample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
        bias: whether to include a bias (i.e. make the layer affine rather than linear).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.
    '''
    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='th',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights

        super(ConvolutionTranspose2D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        if self.bias:
            self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = conv_transpose_out_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_transpose_out_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        if K._BACKEND == 'theano':
            output = self.conv2d_transpose_th(x, self.W, strides=self.subsample,
                              border_mode=self.border_mode,
                              dim_ordering=self.dim_ordering)
        else:
            output = self.conv2d_transpose_tf(x, self.W, strides=self.subsample,
                              border_mode=self.border_mode,
                              dim_ordering=self.dim_ordering)

        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(ConvolutionTranspose2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def conv2d_transpose_tf(self, x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th'):
        if border_mode == 'same':
            padding = 'SAME'
        elif border_mode == 'valid':
            padding = 'VALID'
        else:
            raise Exception('Invalid border mode: ' + str(border_mode))

        strides = (1,) + strides + (1,)
        if K._BACKEND == 'theano':
            input_shape = K.shape(x)
        else:
            try:
                input_shape = K.int_shape(x)
            except Exception as e:
                input_shape = K.shape(x)
        output_shape = self.get_output_shape_for(input_shape)

        if _FLOATX == 'float64':
            # tf conv2d only supports float32
            x = tf.cast(x, 'float32')
            kernel = tf.cast(kernel, 'float32')

        batch_size = tf.shape(x)[0]
        output_shape_tensor = tf.pack([batch_size, output_shape[2], output_shape[3], output_shape[1]])
        if dim_ordering == 'th':
            # TF uses the last dimension as channel dimension,
            # instead of the 2nd one.
            # TH input shape: (samples, input_depth, rows, cols)
            # TF input shape: (samples, rows, cols, input_depth)
            # TH kernel shape: (depth, input_depth, rows, cols)
            # TF kernel shape: (rows, cols, input_depth, depth)
            x = tf.transpose(x, (0, 2, 3, 1))
            kernel = tf.transpose(kernel, (2, 3, 0, 1))
            x = tf.nn.conv2d_transpose(x, kernel, output_shape_tensor, strides, padding=padding)
            x.set_shape((None, output_shape[2], output_shape[3], output_shape[1]))
            x = tf.transpose(x, (0, 3, 1, 2))
            
        elif dim_ordering == 'tf':
            x = tf.nn.conv2d_transpose(x, kernel, output_shape_tensor, strides, padding=padding)
        else:
            raise Exception('Unknown dim_ordering: ' + str(dim_ordering))

        if _FLOATX == 'float64':
            x = tf.cast(x, 'float64')

        return x

    def conv2d_transpose_th(self, x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th'):
        if border_mode == 'same':
            th_border_mode = 'half'
            np_kernel = kernel.eval()
        elif border_mode == 'valid':
            th_border_mode = 'valid'
        else:
            raise Exception('Border mode not supported: ' + str(border_mode))

        input_shape = K.shape(x)
        output_shape = self.get_output_shape_for(input_shape)
        output_shape = (None, ) + output_shape[1:]

        conv_t_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
            x, kernel, output_shape, filter_shape=None, 
            border_mode=th_border_mode, subsample=strides, filter_flip=True)

        if border_mode == 'same':
            if np_kernel.shape[2] % 2 == 0:
                conv_t_out = conv_t_out[:, :, :(x.shape[2] + strides[0] - 1) // strides[0], :]
            if np_kernel.shape[3] % 2 == 0:
                conv_t_out = conv_t_out[:, :, :, :(x.shape[3] + strides[1] - 1) // strides[1]]

        if dim_ordering == 'tf':
            conv_t_out = conv_t_out.dimshuffle((0, 2, 3, 1))

        return conv_t_out
