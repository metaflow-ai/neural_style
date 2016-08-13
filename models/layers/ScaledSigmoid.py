from keras import backend as K
from keras.engine import Layer

class ScaledSigmoid(Layer):
    def __init__(self, scaling=1., **kwargs):
        self.supports_masking = False
        self.scaling = scaling
        super(ScaledSigmoid, self).__init__(**kwargs)


    def call(self, x, mask=None):
        return self.scaling * K.sigmoid(x / self.scaling)

    def get_config(self):
        config = {'scaling': self.scaling}
        base_config = super(ScaledSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))