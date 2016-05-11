import numpy as np 

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

def squared_normalized_euclidian_error(y_true, y_pred):
    loss = K.mean(K.square(y_pred - y_true) / 2.) 

    return loss

#######
# Regularizer
#######
def total_variation_error(y, beta=2.):
    a = K.square(y[:, :, 1:, :-1] - y[:, :, :-1, :-1])
    b = K.square(y[:, :, :-1, 1:] - y[:, :, :-1, :-1])
    loss = K.sum(K.pow(a + b, beta / 2.))

    return loss

##########
# Training
##########
def train_input(input_data, train_iteratee, optimizer, config, max_iter=2000, cross_val_iteratee=None):
    print('Training input')
    losses = {'training_loss': [], 'cross_val_loss': [], 'best_loss': 1e15}
    train_loss = 1e15
    wait = 0
    best_input_data = None
    for i in range(max_iter):
        train_loss, grads_val = train_iteratee([input_data])
        input_data, config = optimizer(input_data, grads_val, config)

        if cross_val_iteratee != None:
            cross_val_loss = cross_val_iteratee([input_data])

        losses.train_loss.append(train_loss)
        if cross_val_iteratee != None:
            losses.cross_val_loss.append(cross_val_loss)

        if i % 1 == 0:
            if cross_val_iteratee != None:
                print(str(i) + ':', train_loss, cross_val_loss)
            else:
                print(str(i) + ':', train_loss)

        if train_loss < losses.best_loss:
            losses.best_loss = train_loss
            best_input_data = np.copy(input_data)
            wait = 0
        else:
            if wait >= 100 and i > max_iter / 2:
                break
            wait +=1

    print("final loss:", losses.best_loss)
    return best_input_data, losses

def train_weights(input_data, trainable_weights, train_iteratee, optimizer, config, max_iter=2000, cross_val_iteratee=None):
    print('Training input')
    losses = {'training_loss': [], 'cross_val_loss': [], 'best_loss': 1e15}
    train_loss = 1e15
    wait = 0
    best_trainable_weights = None
    for i in range(max_iter):
        train_loss, grads_val = train_iteratee([input_data])
        trainable_weights, config = optimizer(trainable_weights, grads_val, config)

        if cross_val_iteratee != None:
            cross_val_loss = cross_val_iteratee([input_data])

        losses.train_loss.append(train_loss)
        if cross_val_iteratee != None:
            losses.cross_val_loss.append(cross_val_loss)

        if i % 1 == 0:
            if cross_val_iteratee != None:
                print(str(i) + ':', train_loss, cross_val_loss)
            else:
                print(str(i) + ':', train_loss)

        if train_loss < losses.best_loss:
            losses.best_loss = train_loss
            best_trainable_weights = np.copy(trainable_weights)
            wait = 0
        else:
            if wait >= 100 and i > max_iter / 2:
                break
            wait +=1

    print("final loss:", losses.best_loss)
    return best_trainable_weights, losses