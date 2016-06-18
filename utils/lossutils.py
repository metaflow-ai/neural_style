import os, re, h5py, math
import numpy as np 

from keras import backend as K
from keras.utils.generic_utils import Progbar
if K._BACKEND == 'theano':
    from theano import tensor as T
else:
    import tensorflow as tf

from utils.imutils import load_image_st
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

def norm_l2(x):
    norm = K.sqrt(K.mean(K.square(x)))
    return x / (norm + K.epsilon())
    
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

def total_variation_error_keras(y, y_fake):
    return total_variation_error(y, 2)
    
##########
# Training
##########
def train_input(input_data, train_iteratee, optimizerName, config={}, max_iter=2000, callbacks=[]):
    losses = {'loss': [], 'val_loss': [], 'best_loss': 1e15}

    wait = 0
    best_input_data = None    
    if optimizerName == 'adam':    
        for i in range(max_iter):
               
            data = train_iteratee([input_data])
            loss = data[0].item(0)
            grads_val = data[1]

            losses['loss'].append(loss)
            if i % 25 == 0:
                print('Iteration: %d/%d' % (i, max_iter) )
                for idx, subloss in enumerate(data):
                    if idx < 2:
                        continue
                    print('    loss %d: %f' % (idx - 1, subloss))
                print('    loss: %f' % (loss))

            if loss < losses['best_loss']:
                losses['best_loss'] = loss
                best_input_data = np.copy(input_data)
                wait = 0
            else:
                if wait >= 100 and i > max_iter / 2:
                    break
                wait +=1

            for callback in callbacks:
                callback({
                    'current_iter':i,
                    'losses': losses,
                    'input_data':input_data
                })

            input_data, config = adam(input_data, grads_val, config)
    else:
        global gogh_inc_val
        gogh_inc_val = 0
        def iter(x):
            global gogh_inc_val
            gogh_inc_val += 1
            x = np.reshape(x, input_data.shape)

            data = train_iteratee([x])
            loss = data[0].item(0)
            grads_val = data[1]
            
            losses['loss'].append(loss)
            if gogh_inc_val % 25 == 0:
                print('Iteration: %d/%d' % (gogh_inc_val, max_iter) )
                for idx, subloss in enumerate(data):
                    if idx < 2:
                        continue
                    print('    loss %d: %f' % (idx - 1, subloss))
                print('    loss: %f' % (loss))

            if loss < losses['best_loss']:
                losses['best_loss'] = loss

            for callback in callbacks:
                callback({
                    'current_iter':gogh_inc_val,
                    'losses': losses,
                    'input_data':input_data
                })

            return loss, grads_val.reshape(-1)

        best_input_data, f ,d = fmin_l_bfgs_b(iter, input_data, factr=1e7, maxiter=max_iter) #factr: 1e7 default value
        best_input_data = np.reshape(best_input_data, input_data.shape)

    print("final loss:", losses['best_loss'])
    return best_input_data, losses

def train_weights(input_dir, size, model, train_iteratee, cv_input_dir=None, max_iter=2000, batch_size=4, callbacks=[], load_result=False):
    losses = {'loss': [], 'val_loss': [], 'best_loss': 1e15}
    
    best_weights = model.get_weights()

    need_more_training = True
    current_iter = 0
    current_epoch = 1
    files = [input_dir + '/' + name for name in os.listdir(input_dir) if len(re.findall('\.(jpe?g|png)$', name))]
    batch_size = min(batch_size, len(files))
    print('total_files %d' % len(files))

    max_epoch = math.floor((batch_size * max_iter) / len(files)) + 1
    while need_more_training:
        print('Epoch %d/%d' % (current_epoch, max_epoch))
        nb_elem = min((max_iter - current_iter) * batch_size, len(files))
        progbar = Progbar(nb_elem)
        progbar_values = []

        ims = []
        y_ims = []
        for idx, fullpath in enumerate(files):
            if load_result == True:
                im, y_im = load_image_st(fullpath, size=size, verbose=False, load_result=load_result) # th ordering, BGR
                y_ims.append(y_im)
            else:
                im = load_image_st(fullpath, size=size, verbose=False, load_result=load_result) # th ordering, BGR
            ims.append(im)

            if len(ims) >= batch_size or idx == len(files) - 1:
                current_iter += 1
                if load_result == True:
                    data = train_iteratee([np.array(ims), np.array(y_ims), True])
                else:
                    data = train_iteratee([np.array(ims), True])
 
                loss = data[0].item(0)
                losses['loss'].append(loss)
                progbar_values.append(('loss', loss))
                for loss_idx, subloss in enumerate(data):
                    if loss_idx < 1:
                        continue
                    progbar_values.append(('loss ' + str(loss_idx), subloss))
                progbar.update(idx + 1, progbar_values)

                if loss < losses['best_loss']:
                    losses['best_loss'] = loss
                    best_weights = model.get_weights()

                for callback in callbacks:
                    callback({
                        current_iter: current_iter,
                        losses: losses,
                        model: model,
                        data: data
                    })

                ims = []
                if current_iter >= max_iter:
                    need_more_training = False
                    break

        current_epoch += 1

    last_weights = model.get_weights()
    print("final best loss:", losses['best_loss'])
    return (best_weights, last_weights), losses 
