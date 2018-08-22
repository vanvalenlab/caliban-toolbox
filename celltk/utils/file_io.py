import os
from os.path import basename, join
import numpy as np
import logging
import tempfile
import shutil
import urllib
from skimage.io import imsave as imsave2
from scipy.ndimage import imread


logger = logging.getLogger(__name__)


def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def imsave(img, output, path, dtype=np.float32):
    """
    save a single image or multiple images.
    len(path) == img.shape[2]
    """
    if isinstance(path, list) or isinstance(path, tuple):
        for num, p in enumerate(path):
            filename = join(output, basename(p).split('.')[0]+'.tif')
            logger.debug('Image (shape {0}) is saved at {1}'.format(img.shape, filename))
            if img.ndim == 2:
                imsave2(filename, img.astype(dtype))
                break
            imsave2(filename, img[:, :, num].astype(dtype))
    else:
        filename = join(output, basename(path).split('.')[0]+'.tif')
        logger.debug('Image (shape {0}) is saved at {1}'.format(img.shape, filename))
        imsave2(filename, img.astype(dtype))


def lbread(path, nonneg=True):
    def uint2int16(img):
        if (img == 0).any():
            return img
        elif img.min() == 32768:
            return img - 32768
    img = imread(path)
    img = uint2int16(img).astype(np.int16)
    if nonneg:
        img[img < 0] = 0
    return img


class LocalPath(object):
    """Generate a local path if URL is given.

    with LocalPath(path) as lpath:
        print(lpath)

    """
    local = True
    def __init__(self, path):
        if os.path.exists(path):
            self.path = path
        else:
            self.local = False
            temp_dir = tempfile.mkdtemp()
            self.path = join(temp_dir, os.path.basename(path))
            urllib.urlretrieve(path, self.path)

    def __enter__(self):
        return self.path

    def __exit__(self, *args):
        if not self.local:
            shutil.rmtree(os.path.dirname(self.path))
