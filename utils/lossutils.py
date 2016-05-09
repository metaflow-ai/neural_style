import numpy as np 

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
    loss = K.sum(K.square(y_pred_grams - y_true_grams))

    return loss

def euclidian_error(y_true, y_pred):
    loss = K.sum(K.square(y_pred - y_true)) / 2

    return loss

def train_on_input(input_data, iteratee, optimizer, config, max_iter=1000):
    print('Training input')
    loss_val = 1e15
    wait = 0
    best_loss = 1e15
    best_input_data = None
    for i in range(max_iter):
        loss_val, grads_val = iteratee([input_data])
        input_data, config = optimizer(input_data, grads_val, config)

        if i % 10 == 0:
            print(str(i) + ':', loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_input_data = np.copy(input_data)
            wait = 0
        else:
            if wait >= 50:
                break
            wait +=1

    print("final loss:", best_loss)
    return best_input_data