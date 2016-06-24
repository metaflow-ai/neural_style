import numpy as np
import os, re, functools
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy import misc
from scipy.misc import imsave

from keras import backend as K

from vgg19.model import VGG_19_mean 

dir = os.path.dirname(os.path.realpath(__file__))
vgg19Dir = dir + '/../vgg19'

# VGG
def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]

    return memoizer

def load_data(absPath, limit=-1, size=(600, 600), verbose=False, st=False):
    X, y = load_images(absPath, limit, size, verbose=verbose, st=st, load_result=True)
    X_cv, y_cv = load_images(absPath + '/cv', limit, size, verbose=verbose, st=st, load_result=True)
        
    return (X, y), (X_cv, y_cv)

def get_image_list(absPath):
    filenames = [absPath + '/' + f for f in os.listdir(absPath) if len(re.findall('\.(jpe?g|png)$', f))]

    return filenames

def load_images(absPath, limit=-1, size=(600, 600), verbose=False, st=False, load_result=False):
    ims = []
    y_ims = []
    if type(absPath) == list:
        fullpaths = absPath
    else:
        fullpaths = get_image_list(absPath)
    for idx, fullpath in enumerate(fullpaths):
        if limit > 0 and idx >= limit:
            break
        
        if st:
            if load_result == True:
                im, y_im = load_image_st(fullpath, size, verbose, load_result)
            else:
                im = load_image_st(fullpath, size, verbose, load_result)
        else:
            if load_result == True:
                im, y_im = load_image(fullpath, size, verbose, load_result)
            else:
                im = load_image(fullpath, size, verbose, load_result)

        ims.append(im)
        if load_result == True:
            y_ims.append(y_im)

    if load_result == True:
        return np.array(ims), np.array(y_ims)
    else:
        return np.array(ims)

def load_image(fullpath, size=(600, 600), verbose=False, load_result=False):
    if verbose:
        print('Loading %s' % fullpath)
    # VGG needs BGR data
    im = misc.imread(fullpath, mode='RGB') # tf ordering
    im = preprocess(im.copy(), size) # tf ordering
    if K.image_dim_ordering() == 'th':
        im = im.transpose(2, 0, 1)

    if load_result == True:
        y_fullpath = get_y_fullpath(fullpath)
        if verbose:
            print('Loading %s' % y_fullpath)

        y_im = misc.imread(y_fullpath, mode='RGB') # height, width, channels
        y_im = preprocess(y_im.copy(), size)
        if K.image_dim_ordering() == 'th':
            y_im = y_im.transpose(2, 0, 1)

        return im, y_im

    return im

def load_image_st(fullpath, size=(600, 600), verbose=False, load_result=False):
    if verbose:
        print('Loading %s' % fullpath)

    perm = np.argsort([2, 1, 0])

    im = misc.imread(fullpath, mode='RGB') # tf ordering, RGB
    im = resize(im, size)
    im = im[:, :, perm] # th ordering, BGR
    if K.image_dim_ordering() == 'th':
        im = im.transpose(2, 0, 1)
    im = im.copy().astype(K.floatx())

    if load_result == True:
        y_fullpath = get_y_fullpath(fullpath)
        if verbose:
            print('Loading %s' % y_fullpath)
        y_im = misc.imread(y_fullpath, mode='RGB') # tf ordering, RGB
        y_im = resize(y_im, size)
        y_im = y_im[:, :, perm] # th ordering, BGR
        if K.image_dim_ordering() == 'th':
            y_im = y_im.transpose(2, 0, 1)
        y_im = y_im.copy().astype(K.floatx())

        return im, y_im

    return im

def get_y_fullpath(x_fullpath):
    splitted_fullpath = x_fullpath.split('/')
    filename = splitted_fullpath.pop()
    y_fullpath = '/'.join(splitted_fullpath + ['results', filename])

    return y_fullpath

def resize(im, size):
    if size == None:
        return im

    nb_dims = len(im.shape)
    if nb_dims == 3:
        if type(size) != tuple:
            size = float(size) / im.shape[0] # make a float
            
        return misc.imresize(im, size, interp='bilinear')
    elif nb_dims == 4:
        if type(size) != tuple:
            size = float(size) / im.shape[1] # make a float

        ims = []
        for i in range(im.shape[0]):                
            new_im = misc.imresize(im[i], size, interp='bilinear')
            ims.append(new_im)

        return np.array(ims) 
    else:
        raise Exception('image should have 3 or 4 dimensions')

# input im is assumed in RGB tf ordering
# return a BGR tf image
def preprocess(im, size=None):
    # im is in a tf ordering
    im = resize(im, size)
    im = im.copy().astype(K.floatx())

    nb_dims = len(im.shape)
    dim_ordering = K.image_dim_ordering()
    mean = load_mean() # load with K.image_dim_ordering()
    if dim_ordering == 'th': # 'th' dim_ordering: [channels, height, width] 
        mean = mean.transpose(0, 2, 3, 1)
    perm = np.argsort([2, 1, 0]) # VGG needs BGR

    if nb_dims == 3:
        im = im[:, :, perm] 
        im -= mean[0]
    elif nb_dims == 4:
        im = im[:, :, :, perm] 
        im -= mean
    else:
        raise Exception('image should have 3 or 4 dimensions')

    return im

def deprocess(im):
    im = np.copy(im)

    nb_dims = len(im.shape)
    mean = load_mean() # load with K.image_dim_ordering()
    perm = np.argsort([2, 1, 0]) # VGG needed BGR
    if nb_dims == 3:
        im += mean[0]
        im = im[:, :, perm]
    elif nb_dims == 4:
        im += mean
        im = im[:, :, :, perm]
    else:
        raise Exception('image should have 3 or 4 dimensions')

    im = im.clip(0, 255)
    im = im.astype('uint8')

    return im

def save_image(fullOutPath, im):
    imsave(fullOutPath, im)

def save_image_st(fullOutPath, im):
    perm = np.argsort([2, 1, 0])

    if K.image_dim_ordering() == 'th':
        im = im.transpose((1, 2, 0))
        
    im = im[:, :, perm]

    im = im.clip(0, 255)
    im = im.astype('uint8')

    imsave(fullOutPath, im)

def load_mean(name='vgg19'):
    if name == 'vgg19':
        return VGG_19_mean(K.image_dim_ordering()) # BGR ordering
    else:
        raise Exception('Invalid mean name:' + name)

def create_noise_tensor(height, width, channels):
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        return np.random.randn(1, height, width, channels).astype(K.floatx()) * 0.001
    elif dim_ordering == 'th':
        return np.random.randn(1, channels, height, width).astype(K.floatx()) * 0.001
    else:
        raise Exception('Invalid dim_ordering value:' + dim_ordering)

def load_hdf5_im(fullpath):
    file = h5py.File(fullpath, 'r')
    dset = file.get('dataset_1')
    im = np.array(dset)
    file.close()

    return im

def dump_as_hdf5(fullpath, data):
    with h5py.File(fullpath, 'w') as hf:
        hf.create_dataset('dataset_1', data=data)

def plot_losses(losses, dir='', prefix='', suffix=''):
    plt.clf()
    if 'val_loss' in losses and len(losses['val_loss']):
        plt.subplot(2, 1, 1)
        plt.plot(losses['loss'])
        plt.title('Training loss')
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')

        plt.subplot(2, 1, 2)
        plt.plot(losses['val_loss'])
        plt.title('Cross validation loss')
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')

        plt.savefig(dir + '/' + prefix + '_val_loss' + suffix + '.png')
        plt.clf()
    else:
        plt.plot(losses['loss'])
        plt.title('Training loss')
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        plt.savefig(dir + '/' + prefix + '_loss' + suffix + '.png')
        plt.clf()
