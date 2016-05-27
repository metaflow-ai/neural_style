import os, re, h5py
import numpy as np 

from keras import backend as K
from keras.utils.generic_utils import Progbar
if K._BACKEND == 'theano':
    from theano import tensor as T
else:
    import tensorflow as tf

from utils.imutils import load_image
from utils.optimizers import adam
from scipy.optimize import fmin_l_bfgs_b

gogh_inc_val = 0

########
# Losses
########
def grams(X, dim_ordering='th'):
    if dim_ordering =='tf':
        X = K.permute_dimensions(X, (0, 3, 1, 2))

    if isinstance(X, (np.ndarray)):
        samples, c, h, w = X.shape 
    elif K._BACKEND == 'theano':
        samples, c, h, w = K.shape(X)
    else:
        try:
            samples, c, h, w = K.int_shape(X)
        except Exception:
            samples, c, h, w = K.shape(X)
        
    X_reshaped = K.reshape(X, (-1, c, h * w))
    X_T = K.permute_dimensions(X_reshaped, (0, 2, 1))
    if K._BACKEND == 'theano':
        X_gram = T.batched_dot(X_reshaped, X_T)
    else:
        X_gram = tf.batch_matmul(X_reshaped, X_T)
    X_gram /= c * h * w

    return X_gram

def frobenius_error(y_true, y_pred):
    loss = K.mean(K.square(y_pred - y_true))

    return loss
    
def load_y_styles(painting_fullpath, layers_name):
    y_styles = []
    with h5py.File(painting_fullpath, 'r') as f:
        for name in layers_name:
            y_styles.append(f[name][()])

    return y_styles



#######
# Regularizer
#######
def total_variation_error(y, beta=1):
    # Negative stop indices are not currently supported in tensorflow ...
    if K._BACKEND == 'theano':
        a = K.square(y[:, :, 1:, :-1] - y[:, :, :-1, :-1])
        b = K.square(y[:, :, :-1, 1:] - y[:, :, :-1, :-1])
    else:
        samples, c, h, w = K.int_shape(y)
        a = K.square(y[:, :, 1:, :w-1] - y[:, :, :h-1, :w-1])
        b = K.square(y[:, :, :h-1, 1:] - y[:, :, :h-1, :w-1])
    if beta == 2:
        loss = K.sum(a + b) / beta
    else:
        loss = K.sum(K.pow(a + b, beta/2.)) / beta

    return loss

##########
# Training
##########
def train_input(input_data, train_iteratee, optimizerName, config={}, max_iter=2000):
    losses = {'training_loss': [], 'cv_loss': [], 'best_loss': 1e15}

    wait = 0
    best_input_data = None
    progbar = Progbar(max_iter)
    progbar_values = []
    if optimizerName == 'adam':    
        for i in range(max_iter):
               
            data = train_iteratee([input_data])
            training_loss = data[0].item(0)
            grads_val = data[1]

            losses['training_loss'].append(training_loss)
            progbar_values.append(('training_loss', training_loss))
            for idx, loss in enumerate(data):
                if idx < 2:
                    continue
                progbar_values.append(('loss' + str(idx), loss))
            progbar.update(i + 1, progbar_values)

            input_data, config = adam(input_data, grads_val, config)

            if training_loss < losses['best_loss']:
                losses['best_loss'] = training_loss
                best_input_data = np.copy(input_data)
                wait = 0
            else:
                if wait >= 100 and i > max_iter / 2:
                    break
                wait +=1
    else:
        global gogh_inc_val
        gogh_inc_val = 0
        def iter(x):
            global gogh_inc_val
            gogh_inc_val += 1
            x = np.reshape(x, input_data.shape)

            data = train_iteratee([x])
            training_loss = data[0].item(0)
            grads_val = data[1]
            
            losses['training_loss'].append(training_loss)
            progbar_values.append(('training_loss', training_loss))
            for idx, loss in enumerate(data):
                if idx < 2:
                    continue
                progbar_values.append(('loss' + str(idx), loss))
            progbar.update(gogh_inc_val, progbar_values)

            if training_loss < losses['best_loss']:
                losses['best_loss'] = training_loss

            return training_loss, grads_val.reshape(-1)

        best_input_data, f ,d = fmin_l_bfgs_b(iter, input_data, maxiter=max_iter)
        best_input_data = np.reshape(best_input_data, input_data.shape)

    print("final loss:", losses['best_loss'])
    return best_input_data, losses

def train_weights(input_dir, size, model, train_iteratee, cv_input_dir=None, max_iter=2000, batch_size=4):
    losses = {'training_loss': [], 'cv_loss': [], 'best_loss': 1e15}
    
    best_trainable_weights = model.get_weights()

    need_more_training = True
    wait = 0
    current_iter = 0
    current_epoch = 0
    files = [input_dir + '/' + name for name in os.listdir(input_dir) if len(re.findall('\.(jpg|png)$', name))]
    
    while need_more_training:
        print('Epoch %d, max_iter %d, total_files %d' % (current_epoch + 1, max_iter, len(files)))
        progbar = Progbar(max_iter)
        progbar_values = []

        ims = []
        current_batch = 0
        for idx, fullpath in enumerate(files):
            im = load_image(fullpath, size=size)
            ims.append(im)
            if len(ims) >= batch_size:
                ims = np.array(ims)
                training_loss = train_iteratee([ims, True])
                training_loss = training_loss[0].item(0)
                losses['training_loss'].append(training_loss)
                progbar_values.append(('training_loss', training_loss))
                if cv_input_dir != None:
                    cv_loss = train_iteratee([cv_input_dir, False])
                    cv_loss = cv_loss[0].item(0)
                    losses['cv_loss'].append(cv_loss)
                    progbar_values.append(('cv_loss', cv_loss))

                progbar.update(current_iter, progbar_values)

                if training_loss < losses['best_loss']:
                    losses['best_loss'] = training_loss
                    best_trainable_weights = model.get_weights()
                    wait = 0
                else:
                    if wait >= 100 and current_iter > max_iter / 2:
                        need_more_training = False
                        break
                    wait +=1

                current_iter += 1
                current_batch += 1
                ims = []

                if current_iter >= max_iter:
                    need_more_training = False
                    break

        current_epoch += 1

    print("final best loss:", losses['best_loss'])
    return best_trainable_weights, losses
