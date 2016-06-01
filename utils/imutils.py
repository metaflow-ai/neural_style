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

def load_images(absPath, limit=-1, size=(600, 600), dim_ordering='tf', verbose=False, st=False):
    ims = []
    filenames = [f for f in os.listdir(absPath) if len(re.findall('\.(jpe?g|png)$', f))]
    for idx, filename in enumerate(filenames):
        if limit > 0 and idx >= limit:
            break
        
        fullpath = absPath + '/' + filename
        if st:
            im = load_image_st(fullpath, size, verbose)
        else:
            im = load_image(fullpath, size, dim_ordering, verbose)
        ims.append(im)

    return np.array(ims)

def load_image(fullpath, size=(600, 600), dim_ordering='tf', verbose=False):
    if verbose:
        print('Loading ' + fullpath)
    # VGG needs BGR data
    im = misc.imread(fullpath, mode='RGB') # height, width, channels
    im = preprocess(im.copy(), size)

    if dim_ordering == 'th':
        im = im.transpose(2, 0, 1)

    return im

def load_image_st(fullpath, size=(600, 600), verbose=False):
    if verbose:
        print('Loading ' + fullpath)

    im = misc.imread(fullpath, mode='RGB') # tf ordering, RGB
    if size != None:
        im = misc.imresize(im, size, interp='bilinear')
    perm = np.argsort([2, 1, 0])
    im = im[:, :, perm] # th ordering, BGR
    im = im.transpose(2, 0, 1) # th ordering, BGR

    return im.copy().astype(K.floatx())


# im should be in RGB order
# dim_ordering: How "im" is ordered
def preprocess(im, size=None, dim_ordering='tf'):
    if size != None:
        im = misc.imresize(im, size, interp='bilinear')
    im = im.copy().astype(K.floatx())

    nb_dims = len(im.shape)

    if dim_ordering == 'th': # 'th' dim_ordering: [channels, height, width] 
        # tf order
        if nb_dims == 3:
            im = im.transpose(1, 2, 0)
        else:
            im = im.transpose(0, 2, 3, 1)

    # VGG needs BGR
    mean = load_mean() # vgg19, BGR, tf ordering
    perm = np.argsort([2, 1, 0])
    if nb_dims == 3:
        im = im[:, :, perm] 
        im -= mean[0]
    elif nb_dims == 4:
        im = im[:, :, :, perm] 
        im -= mean
    else:
        raise Exception('image should have 3 or 4 dimensions')

    return im

def deprocess(im, dim_ordering='tf'):
    im = np.copy(im)

    if dim_ordering == 'th':
        im = im.transpose((1, 2, 0))

    # Back to RGB
    perm = np.argsort([2, 1, 0])
    nb_dims = len(im.shape)
    if nb_dims == 3:
        im += load_mean()[0]
        im = im[:, :, perm]
    elif nb_dims == 4:
        im += load_mean()
        im = im[:, :, :, perm]
    else:
        raise Exception('image should have 3 or 4 dimensions')

    im = im.clip(0, 255)
    im = im.astype('uint8')

    return im

def save_image(fullOutPath, im):
    imsave(fullOutPath, im)

def save_image_st(fullOutPath, im):
    im = im.transpose((1, 2, 0))
    
    perm = np.argsort([2, 1, 0])
    im = im[:, :, perm]

    im = im.clip(0, 255)
    im = im.astype('uint8')

    imsave(fullOutPath, im)

@memoize
def load_mean(name='vgg19', dim_ordering='tf'):
    if name == 'vgg19':
        return VGG_19_mean(dim_ordering) # BGR ordering
    else:
        raise Exception('Invalid mean name:' + name)

def create_noise_tensor(height, width, channels, dim_ordering='tf'):
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
    if len(losses['cv_loss']):
        plt.subplot(2, 1, 1)
        plt.plot(losses['training_loss'])
        plt.title('Training loss')
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')

        plt.subplot(2, 1, 2)
        plt.plot(losses['cv_loss'])
        plt.title('Cross validation loss')
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')

        plt.savefig(dir + '/' + prefix + 'cv_loss' + suffix + '.png')
        plt.clf()
    else:
        plt.plot(losses['training_loss'])
        plt.title('Training loss')
        plt.xlabel('Iteration number')
        plt.ylabel('Loss value')
        plt.savefig(dir + '/' + prefix + 'training_loss' + suffix + '.png')
        plt.clf()
