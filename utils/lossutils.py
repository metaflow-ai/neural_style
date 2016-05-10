import numpy as np 
import scipy

from keras.callbacks import Callback
from keras import backend as K

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss').item(0))

########
# Losses
########
def grams_frobenius_error(y_true, y_pred):
    samples, c, h, w = y_true.shape

    # Compute the grams matrix
    y_true_reshaped = K.reshape(y_true, (samples, c, h * w))
    y_pred_reshaped = K.reshape(y_pred, (samples, c, h * w))
    y_true_T = K.permute_dimensions(y_true_reshaped, (0, 2, 1))
    y_pred_T = K.permute_dimensions(y_pred_reshaped, (0, 2, 1))
    y_true_grams = K.dot(y_true_reshaped, y_true_T) / (2. * c * h * w)
    y_pred_grams = K.dot(y_pred_reshaped, y_pred_T) / (2. * c * h * w)

    # Compute the frobenius norm
    loss = K.sum(K.square(y_pred_grams - y_true_grams))

    return loss

def squared_nornalized_euclidian_error(y_true, y_pred):
    samples, c, h, w = y_true.shape

    # Compute the euclidian distance
    loss = K.sum(K.square(y_pred - y_true)) / (2. * c * h * w)

    return loss

#######
# Regularizer
#######
def total_variation_error(y, beta=2.):
    samples, c, h, w = y.shape

    a = K.square(y[:, :, 1:, :-1] - y[:, :, :-1, :-1])
    b = K.square(y[:, :, :-1, 1:] - y[:, :, :-1, :-1])
    loss = K.sum(K.pow(a + b, beta / 2.))

    return loss

##########
# Training
##########
def train_on_input(input_data, iteratee, optimizer, config, max_iter=2000):
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
            if wait >= 100 and i > max_iter / 2:
                break
            wait +=1

    print("final loss:", best_loss)
    return best_input_data