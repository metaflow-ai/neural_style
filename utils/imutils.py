import numpy as np

import os, re
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
    im -= im.mean()
    im /= (im.std() + 1e-5)
    im *= 0.1

    im += 0.5
    im = np.clip(im, 0, 1)

    # convert to RGB array
    im *= 255
    im = im.transpose((1, 2, 0))
    im = np.clip(im, 0, 255).astype('uint8')

    imsave(fullOutPath, im)