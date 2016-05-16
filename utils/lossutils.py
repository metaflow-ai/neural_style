import numpy as np 

from keras import backend as K

########
# Losses
########
def grams_frobenius_error(y_true, y_pred):
    samples_true, c, h, w = y_true.shape
    samples_preds = y_pred.shape[0]

    # Compute the grams matrix
    y_true_reshaped = K.reshape(y_true, (samples_true, c, h * w))
    y_pred_reshaped = K.reshape(y_pred, (samples_preds, c, h * w))
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
def train_input(input_data, train_iteratee, optimizer, config={}, cv_input_data=None, cross_val_iteratee=None, max_iter=2000):
    print('Training input')
    losses = {'training_loss': [], 'cv_loss': [], 'best_loss': 1e15}
    training_loss = 1e15
    wait = 0
    best_input_data = None
    for i in range(max_iter):
        training_loss, grads_val = train_iteratee([input_data])
        input_data, config = optimizer(input_data, grads_val, config)

        if cross_val_iteratee != None:
            cv_loss = cross_val_iteratee([input_data])

        losses['training_loss'].append(training_loss)
        if cross_val_iteratee != None:
            losses['cv_loss'].append(cv_loss)

        if i % 10 == 0:
            if cross_val_iteratee != None:
                print(str(i) + ':', training_loss, cv_loss)
            else:
                print(str(i) + ':', training_loss)

        if training_loss < losses['best_loss']:
            losses['best_loss'] = training_loss
            best_input_data = np.copy(input_data)
            wait = 0
        else:
            if wait >= 100 and i > max_iter / 2:
                break
            wait +=1

    print("final loss:", losses['best_loss'])
    return best_input_data, losses

def train_weights(train_input_data, st_model, train_iteratee, batch=32, cv_input_data=None, cross_val_iteratee=None, max_iter=2000):
    print('Training input')
    losses = {'training_loss': [], 'cv_loss': [], 'best_loss': 1e15}
    wait = 0
    best_trainable_weights = st_model.get_weights()
    for i in range(max_iter):
        training_loss = train_iteratee([train_input_data, True])
        training_loss = training_loss[0].item(0)
        if cross_val_iteratee != None:
            cv_loss = cross_val_iteratee([cv_input_data, False])
            cv_loss = cv_loss[0].item(0)

        losses['training_loss'].append(training_loss)
        if cross_val_iteratee != None and not np.isinf(cv_loss) and not np.isnan(cv_loss):
            losses['cv_loss'].append(cv_loss)

        if i % 1 == 0:
            if cross_val_iteratee != None:
                print(str(i) + ':', training_loss, cv_loss)
            else:
                print(str(i) + ':', training_loss)
        
        if training_loss < losses['best_loss']:
            losses['best_loss'] = training_loss
            best_trainable_weights = st_model.get_weights()
            wait = 0
        else:
            if wait >= 100 and i > max_iter / 2:
                break
            wait +=1

    print("final best loss:", losses['best_loss'])
    return best_trainable_weights, losses
