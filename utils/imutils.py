import numpy as np
import os, re
from scipy import misc, ndimage as ndi

def load_images(absPath, limit=-1):
    ims = []
    files = [f for f in os.listdir(absPath) if len(re.findall('\.jpg$', f))]
    for idx, filename in enumerate(files):
        if limit > 0 and idx >= limit:
            break
        fullpath = absPath + '/' + filename
        im = ndi.imread(fullpath, mode='RGB')
        im = misc.imresize(im, (256,256)).transpose(2, 0, 1).astype(np.float32)
        ims.append(im)

    return np.array(ims)

def load_image(fullpath, limit=-1):
    im = ndi.imread(fullpath, mode='RGB')
    im = misc.imresize(im, (256,256)).transpose(2, 0, 1).astype(np.float32)

    return np.array(im)

def load_data(dataPath):
    return ''