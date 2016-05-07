from keras.callbacks import Callback
from keras import backend as K

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss').item(0))


def grams_frobenius_error(y_true, y_pred):
    samples, c, h, w = y_true.shape

    y_true_reshaped = K.reshape(y_true, (samples, c, h * w))
    y_pred_reshaped = K.reshape(y_pred, (samples, c, h * w))
    y_true_T = K.permute_dimensions(y_true_reshaped, (0, 2, 1))
    y_pred_T = K.permute_dimensions(y_pred_reshaped, (0, 2, 1))
    y_true_grams = K.dot(y_true_reshaped, y_true_T) / (2 * c * h * w)
    y_pred_grams = K.dot(y_pred_reshaped, y_pred_T) / (2 * c * h * w)
    loss = K.sum(K.square(y_true_grams - y_pred_grams))

    return loss

def euclidian_error(y_true, y_pred):
    samples, c, h, w = y_true.shape

    loss = K.sum(K.square(y_pred - y_true)) / (2 * c * h * w)

    return loss
