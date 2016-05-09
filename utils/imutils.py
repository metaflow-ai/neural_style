import numpy as np

import os, re
import h5py
from scipy import misc, ndimage as ndi
from scipy.misc import imsave

def load_images(absPath, limit=-1):
    ims = []
    files = [f for f in os.listdir(absPath) if len(re.findall('\.(jpg|png)$', f))]
    for idx, filename in enumerate(files):
        if limit > 0 and idx >= limit:
            break
        fullpath = absPath + '/' + filename
        im = ndi.imread(fullpath, mode='RGB')
        im = misc.imresize(im, (256,256)).transpose(2, 0, 1).astype(np.float32)
        ims.append(im)

    return np.array(ims)

def load_image(fullpath):
    im = ndi.imread(fullpath, mode='RGB')
    im = misc.imresize(im, (256,256)).transpose(2, 0, 1).astype(np.float32)

    return np.array(im)

def load_data(dataPath):
    return ''

def create_noise_tensor(channels, width, height):
    return np.random.rand(1, channels, width, height) * 20 + 128.

def deprocess_image(im, fullOutPath):
    im = np.copy(im)
    im -= im.min() # [0, +Infinity]
    im /= im.max() # [0, 1]

    # convert to RGB array
    im *= 255
    im = im.transpose((1, 2, 0))
    im = np.clip(im, 0, 255).astype('uint8')

    imsave(fullOutPath, im)

def dump_as_hdf5(data, fullpath):
    with h5py.File(fullpath, 'w') as hf:
        hf.create_dataset('dataset_1', data=data)


# dir = os.path.dirname(os.path.realpath(__file__))
# resultsDir = dir + '/../models/results/vgg16'
# layer_name = 'conv_1_1'
# with h5py.File(resultsDir + '/feat_' + layer_name + '.hdf5','r') as hf:
#     print('List of arrays in this file: \n', hf.keys())
#     data = hf.get('dataset_1')
#     fullOutPath = resultsDir + '/feat_' + layer_name + ".png"
#     deprocess_image(data, fullOutPath)
