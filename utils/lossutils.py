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

##########
# Training
##########
def train_input(input_data, train_iteratee, optimizerName, config={}, max_iter=2000):
    losses = {'training_loss': [], 'cv_loss': [], 'best_loss': 1e15}

    wait = 0
    best_input_data = None    
    if optimizerName == 'adam':    
        for i in range(max_iter):
               
            data = train_iteratee([input_data])
            training_loss = data[0].item(0)
            grads_val = data[1]

            losses['training_loss'].append(training_loss)
            if i % 25 == 0:
                print('Iteration: %d/%d' % (i, max_iter) )
                for idx, loss in enumerate(data):
                    if idx < 2:
                        continue
                    print('    loss %d: %f' % (idx - 1, loss))
                print('    training_loss: %f' % (training_loss))

            if training_loss < losses['best_loss']:
                losses['best_loss'] = training_loss
                best_input_data = np.copy(input_data)
                wait = 0
            else:
                if wait >= 100 and i > max_iter / 2:
                    break
                wait +=1

            input_data, config = adam(input_data, grads_val, config)
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
            if gogh_inc_val % 25 == 0:
                print('Iteration: %d/%d' % (gogh_inc_val, max_iter) )
                for idx, loss in enumerate(data):
                    if idx < 2:
                        continue
                    print('    loss %d: %f' % (idx - 1, loss))
                print('    training_loss: %f' % (training_loss))

            if training_loss < losses['best_loss']:
                losses['best_loss'] = training_loss

            return training_loss, grads_val.reshape(-1)

        best_input_data, f ,d = fmin_l_bfgs_b(iter, input_data, maxiter=max_iter)
        best_input_data = np.reshape(best_input_data, input_data.shape)

    print("final loss:", losses['best_loss'])
    return best_input_data, losses

def train_weights(input_dir, size, model, train_iteratee, cv_input_dir=None, max_iter=2000, batch_size=4, callbacks=[]):
    losses = {'training_loss': [], 'cv_loss': [], 'best_loss': 1e15}
    
    best_trainable_weights = model.get_weights()

    need_more_training = True
    current_iter = 0
    current_epoch = 0
    files = [input_dir + '/' + name for name in os.listdir(input_dir) if len(re.findall('\.(jpe?g|png)$', name))]
    
    while need_more_training:
        print('Epoch %d, max_iter %d, total_files %d' % (current_epoch + 1, max_iter, len(files)))
        progbar = Progbar(max_iter)
        progbar_values = []

        ims = []
        for idx, fullpath in enumerate(files):
            im = load_image(fullpath, size=size, dim_ordering='th')
            ims.append(im)
            if len(ims) >= batch_size:
                current_iter += 1
                data = train_iteratee([np.array(ims), True])

                training_loss = data[0].item(0)
                losses['training_loss'].append(training_loss)
                progbar_values.append(('training_loss', training_loss))
                for loss_idx, loss in enumerate(data):
                    if idx < 1:
                        continue
                    progbar_values.append(('loss ' + str(loss_idx), loss))
                # if cv_input_dir != None:
                #     data_loss = train_iteratee([cv_input_dir, False])
                #     cv_loss = data_loss[0].item(0)
                #     losses['cv_loss'].append(cv_loss)
                #     progbar_values.append(('cv_loss', cv_loss))

                progbar.update(current_iter, progbar_values)

                if training_loss < losses['best_loss']:
                    losses['best_loss'] = training_loss
                    best_trainable_weights = model.get_weights()

                for tuple in callbacks:
                    if current_iter % tuple[0] == 0:
                        tuple[1]({
                            current_iter:current_iter,
                            losses: losses,
                            model: model
                        })
                ims = []
                if current_iter >= max_iter:
                    need_more_training = False
                    break

        current_epoch += 1

    print("final best loss:", losses['best_loss'])
    return best_trainable_weights, losses
