from keras.callbacks import Callback
from keras import backend as K

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss').item(0))


def grams_frobenius_error(y_true, y_pred):
    eps = 1e-8
    y_true_T = K.permute_dimensions(y_true, (0, 2, 1))
    y_pred_T = K.permute_dimensions(y_pred, (0, 2, 1))
    y_true_grams = K.dot(y_true, y_true_T) / (y_true.shape[1] * y_true.shape[2])
    y_pred_grams = K.dot(y_pred, y_pred_T) / (y_pred.shape[1] * y_pred.shape[2])
    return K.sum(K.max(K.square(y_true_grams - y_pred_grams), eps))

def euclidian_error(y_true, y_pred):
    eps = 1e-8
    return K.sum(K.max(K.square(y_pred - y_true), eps)) / (2 * y_true.shape[1] * y_true.shape[2] * y_true.shape[3])