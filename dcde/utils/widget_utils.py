# import statements
from __future__ import absolute_import

from io import BytesIO

from IPython.display import Image
from ipywidgets import interact, interactive, fixed
import matplotlib as mpl
from skimage import data, filters, io, img_as_uint
import numpy as np
import skimage as sk
import os
from scipy import ndimage
import scipy
import sys
from ipywidgets import interact
from skimage.io import imread
import matplotlib.pyplot as plt

import scipy.ndimage as ndi



def arr2img(arr):
    """Display a 2- or 3-d numpy array as an image."""
    if arr.ndim == 2:
        format, cmap = 'png', mpl.cm.gray
    elif arr.ndim == 3:
        format, cmap = 'jpg', None
    else:
        raise ValueError("Only 2- or 3-d arrays can be displayed as images.")
    # Don't let matplotlib autoscale the color range so we can control overall luminosity
    vmax = 255 if arr.dtype == 'uint8' else 1.0
    with BytesIO() as buffer:
        mpl.image.imsave(buffer, arr, format=format, cmap=cmap)
        out = buffer.getvalue()
    return Image(out)

def choose_img(name, dirpath):
    """Used to choose which image we want to use for the widget tester"""
    global img
    filepath = os.path.join(dirpath, name)
    img = img_as_uint(imread(filepath))
    return arr2img(img)

def edit_image(image, sigma=1.0, equalize_hist=False, equalize_adapthist=False, gamma_adjust = 1.0):
    """Used to edit the image using the widget tester"""
    global hist
    global adapthist
    global gaussian_sigma
    global gamma
    
    new_image = filters.gaussian(image, sigma=sigma, multichannel=False)
    new_image += 1000*sk.filters.sobel(new_image)
    new_image = sk.exposure.adjust_gamma(new_image, gamma_adjust, gain = 1)
    new_image[:] = -1.0*new_image[:]
    new_image=sk.exposure.rescale_intensity(new_image, in_range = 'image', out_range = 'float')
    
    if(equalize_hist == True):
        #new_image=sk.exposure.rescale_intensity(new_image, in_range = 'image', out_range = 'np.uint16')
        new_image = sk.exposure.equalize_hist(new_image, nbins=256, mask=None)
        
    if(equalize_adapthist == True):
        new_image = sk.exposure.equalize_adapthist(new_image, kernel_size=None, clip_limit=0.01, nbins=256)
     
    new_image = sk.exposure.rescale_intensity(new_image, in_range = 'image', out_range = np.uint8)
    new_image = new_image.astype(np.uint8)
    
    hist = equalize_hist
    adapthist = equalize_adapthist
    gaussian_sigma = sigma
    gamma = gamma_adjust

    
    return arr2img(new_image)
