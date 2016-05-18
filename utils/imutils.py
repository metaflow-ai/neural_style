import numpy as np
import os, re
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy import misc, ndimage as ndi
from scipy.misc import imsave

from keras.backend.common import _FLOATX

def load_images(absPath, limit=-1, size=None, dim_ordering='th'):
    ims = []
    filenames = [f for f in os.listdir(absPath) if len(re.findall('\.(jpg|png)$', f))]
    for idx, filename in enumerate(filenames):
        if limit > 0 and idx >= limit:
            break
        fullpath = absPath + '/' + filename
        im = load_image(fullpath, size, dim_ordering)
        ims.append(im)

    return np.array(ims)

def load_image(fullpath, size=None, dim_ordering='th'):
    im = ndi.imread(fullpath, mode='RGB') # height, width, channels
    if size != None:
        im = misc.imresize(im, size)
    if dim_ordering == 'th':
        # for 'th' dim_ordering (default): [batch, channels, height, width] 
        im = im.transpose(2, 0, 1)
    im = im.astype(_FLOATX)

    return np.array(im)

def load_data(dataPath):
    return ''

def create_noise_tensor(channels, width, height):
    return np.random.rand(1, channels, width, height) * 20 + 128.

def deprocess_image(fullOutPath, im, normalize=True):
    im = np.copy(im)
    if normalize:
        im -= im.min() # [0, +Infinity]
        im /= im.max() # [0, 1]
        im *= 255
    im = im.transpose((1, 2, 0))
    im = np.clip(im, 0, 255).astype('uint8')

    imsave(fullOutPath, im)

def dump_as_hdf5(data, fullpath):
    with h5py.File(fullpath, 'w') as hf:
        hf.create_dataset('dataset_1', data=data)

def plot_losses(losses, dir='', prefix='', suffix=''):
    plt.plot(losses['training_loss'])
    plt.title('Training loss')
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.savefig(dir + '/' + prefix + 'training_loss' + suffix + '.png')

    plt.plot(losses['cv_loss'])
    plt.title('Cross validation loss')
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.savefig(dir + '/' + prefix + 'cv_loss' + suffix + '.png')

# dir = os.path.dirname(os.path.realpath(__file__))
# resultsDir = dir + '/../models/results/vgg16'
# layer_name = 'conv_1_1'
# with h5py.File(resultsDir + '/feat_' + layer_name + '.hdf5','r') as hf:
#     print('List of arrays in this file: \n', hf.keys())
#     data = hf.get('dataset_1')
#     fullOutPath = resultsDir + '/feat_' + layer_name + ".png"
#     deprocess_image(data, fullOutPath)
