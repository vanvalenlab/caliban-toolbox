import numpy as np
from scipy.ndimage import imread as imread0
from skimage.io import imread


def imread_check_tiff(path):
    img = imread0(path)
    if img.dtype == 'object':
        img = tiff.imread(path)
    return img


def imread(path):
    if isinstance(path, tuple) or isinstance(path, list):
        st = []
        for p in path:
            st.append(imread_check_tiff(p))
        img = np.dstack(st)
        if img.shape[2] == 1:
            np.squeeze(img, axis=2)
        return img
    else:
        return imread_check_tiff(path)
