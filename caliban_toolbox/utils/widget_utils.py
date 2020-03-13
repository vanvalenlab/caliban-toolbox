# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/caliban-toolbox/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''
Functions to select and adjust images with widgets in jupyter notebooks
'''

# import statements
from __future__ import absolute_import

from ipywidgets import interact, interactive, fixed

from skimage import filters, img_as_uint
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import os
from imageio import imread

from caliban_toolbox.utils.io_utils import get_img_names

def choose_img(name, dirpath):
    '''
    Pick an image from a directory to see effects of contrast adjustment steps; image chosen is displayed
    
    Args:
        name: name of image file
        dirpath: full path to directory containing images
    
    Returns:
        Full path to selected image
    '''
    filepath = os.path.join(dirpath, name)
    img = imread(filepath)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img, cmap = mpl.cm.gray)
    return filepath

def choose_img_pair(frame, raw_dir, overlay_dir):
    '''
    Pick two paired images from two directories (eg, each directory contains a different channel)
    to see effects of contrast adjustment steps; both images chosen are displayed
    
    Args:
        frame: index of image location in directory
        raw_dir: full path to first image-containing directory
        overlay_dir: full path to second image-containing directory
        
    Returns:
        Full paths to both images selected 
    '''
    #load raw and overlay images based on frame number
    raw_img_name = get_img_names(raw_dir)[frame]
    raw_img_path = os.path.join(raw_dir, raw_img_name)
    raw_img = imread(raw_img_path)
    
    overlay_img_name = get_img_names(overlay_dir)[frame]
    overlay_img_path = os.path.join(overlay_dir, overlay_img_name)
    overlay_img = imread(overlay_img_path)
    
    fig, ax = plt.subplots(figsize=(16, 12), nrows = 2)
    
    plt.subplot(211)
    plt.imshow(raw_img, cmap = mpl.cm.gray)
    plt.subplot(212)
    plt.imshow(overlay_img, cmap = mpl.cm.gray)

    plt.show()
    
    return raw_img_path, overlay_img_path
    

def adjust_image_interactive(image, blur=1.0, sobel_toggle=True, sobel_factor=100, invert_img=True, gamma_adjust=1.0,
               equalize_hist=False, equalize_adapthist=False, v_min=0, v_max=255):
    '''
    Display effects of contrast adjustment on an image
    
    Args:
        image: image to adjust, as array
        blur: how much to blur image with gaussian filter (values between 0 and 1 sharpen image)
        sobel_toggle: whether to apply a sobel filter to the image (find edges of objects)
        sobel_factor: how heavily the sobel filter is applied
        invert_img: whether to invert light and dark in the image
        gamma_adjust: how much to adjust the overall brightness of the image
        equalize_hist: whether to use histogram equalization on the image
        equalize_adapthist: whether to use adaptive histogram equilazation on the image
        v_min: minimum value from image to be rescaled, pixels with intensities below this value will be set to zero
        v_max: maximum value from image to be rescaled, pixels with intensities above this value will be set to 255
    
    Returns:
        Contrast-adjusted image
    '''

    new_image = filters.gaussian(image, sigma=blur, multichannel=False)
    
    if sobel_toggle:
        new_image += sobel_factor * filters.sobel(new_image)
        
    new_image = sk.exposure.adjust_gamma(new_image, gamma_adjust, gain = 1)
    
    if invert_img:
        new_image = sk.util.invert(new_image)
        
    new_image=sk.exposure.rescale_intensity(new_image, in_range = 'image', out_range = 'float')
    
    if(equalize_hist == True):
        #new_image=sk.exposure.rescale_intensity(new_image, in_range = 'image', out_range = 'np.uint16')
        new_image = sk.exposure.equalize_hist(new_image, nbins=256, mask=None)
        
    if(equalize_adapthist == True):
        new_image = sk.exposure.equalize_adapthist(new_image, kernel_size=None, clip_limit=0.01, nbins=256)
     
    new_image = sk.exposure.rescale_intensity(new_image, in_range = 'image', out_range = np.uint8)
    new_image = new_image.astype(np.uint8)

    #adjust min/max of image after it is rescaled to np.uint8
    new_image = sk.exposure.rescale_intensity(new_image, in_range=(v_min, v_max))
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(new_image, cmap = mpl.cm.gray)

    return new_image


def adjust_image(image, adjust_kwargs):
    '''
    Contrast adjusts an image using assortment of filters and skimage functions

    Args:
        image: image to adjust, as array, with dtype np.float32
        sigma: how much to blur image with gaussian filter (values between 0 and 1 sharpen image)
        hist: whether to use histogram equalization on the image
        adapthist: whether to use adaptive histogram equilazation on the image
        gamma: how much to adjust the overall brightness of the image
        sobel_option: whether to apply a sobel filter to the image (find edges of objects)
        sobel: how heavily the sobel filter is applied
        invert: whether to invert light and dark in the image
        v_min: minimum value from image to be rescaled, pixels with intensities below this value will be set to zero
        v_max: maximum value from image to be rescaled, pixels with intensities above this value will be set to 255

    Returns:
        Contrast adjusted image as a numpy array (np.uint8)
    '''

    if len(image.shape) > 2:
        print("Too many dimensions in your image. Make sure to split out channels and don't feed in image stacks")
        return

    # get values
    sigma = adjust_kwargs['blur']
    hist = adjust_kwargs['equalize_hist']
    adapthist = adjust_kwargs['equalize_adapthist']
    gamma = adjust_kwargs['gamma_adjust']
    sobel_option = adjust_kwargs['sobel_toggle']
    sobel = adjust_kwargs['sobel_factor']
    invert = adjust_kwargs['invert_img']
    v_min = adjust_kwargs['v_min']
    v_max = adjust_kwargs['v_max']

    # Blur
    image = filters.gaussian(image, sigma, multichannel=False)

    # Find edges
    if sobel_option:
        image = sk.exposure.rescale_intensity(image, in_range='image', out_range='float')
        image += sobel * sk.filters.sobel(image)

    # Adjust gamma
    image = sk.exposure.adjust_gamma(image, gamma, gain=1)

    # Invert
    if invert:
        image = sk.util.invert(image)

    if (hist):
        image = sk.exposure.equalize_hist(image, nbins=256, mask=None)

    if (adapthist):
        image = sk.exposure.rescale_intensity(image, in_range='image', out_range='float')
        image = sk.exposure.equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256)

    # Rescale intensity
    image = sk.exposure.rescale_intensity(image, in_range='image', out_range=np.uint8)
    image = image.astype(np.uint8)
    # okay to lose precision in these images--they don't get used in training data, just annotation

    image = sk.exposure.rescale_intensity(image, in_range=(v_min, v_max))

    return image


def overlay_images_interactive(img_1, img_2, prop_img_1, v_min=0, v_max=255):
    '''
    Display effects of overlaying two contrast-adjusted images
    
    Args:
        img_1: first image of pair to form composite, as numpy array
        img_2: second image of pair to form composite, as numpy array
        prop_img_1: what percentage of the composite image comes from the first input image
        v_min: minimum value from image to be rescaled, pixels with intensities below this value will be set to zero
        v_max: maximum value from image to be rescaled, pixels with intensities above this value will be set to 255
        
    Returns:
        None
    '''
    
    prop_img_2 = 1.0 - prop_img_1
    mod_img = img_1 * prop_img_1 + img_2 * prop_img_2

    # mod_img is currently a float, but will cause errors because values are not between 0 and 1
    mod_img = mod_img.astype(np.uint8)
    
    mod_img = sk.exposure.equalize_adapthist(mod_img, kernel_size=None, clip_limit=0.01, nbins=256)
    
    # equalize_adapthist outputs float64 image
    # rescale image to (0,255) before changing to uint8 dtype
    mod_img = sk.exposure.rescale_intensity(mod_img, in_range="image", out_range=np.uint8)
    mod_img = mod_img.astype(np.uint8)
    
    # adjust brightness settings using v_min and v_max in range of uint8
    mod_img = sk.exposure.rescale_intensity(mod_img, in_range=(v_min, v_max))    
    
    # show modified image
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(mod_img, cmap = mpl.cm.gray)
    
    return None


def overlay_images(img_1, img_2, prop_img_1, v_min, v_max):
    """Takes two images and overlays them on top of each other

    Inputs
        img_1: first image to be overlaid
        img_2: second image to be overlaid
        prop_img_1: proportion of img_1 to be included in overlay
        v_min: minimum value from image for rescaling, pixles below this value will be set to 0
        v_max: maximum value from image for rescaling, pixels above this value will be set to 255"""

    prop_img_2 = 1.0 - prop_img_1
    mod_img = img_1 * prop_img_1 + img_2 * prop_img_2
    mod_img = sk.exposure.rescale_intensity(mod_img, in_range='image', out_range='uint8')
    mod_img = mod_img.astype(np.uint8)
    mod_img = sk.exposure.equalize_adapthist(mod_img, kernel_size=None, clip_limit=0.01, nbins=256)

    # equalize_adapthist outputs float64 image
    # rescale image to (0,255) before changing to uint8 dtype
    mod_img = sk.exposure.rescale_intensity(mod_img, in_range="image", out_range=np.uint8)
    mod_img = mod_img.astype(np.uint8)

    # rescale brightness to user-defined range
    mod_img = sk.exposure.rescale_intensity(mod_img, in_range=(v_min, v_max))
    return mod_img


def choose_img_from_stack(stack, slice_index, chan_name):
    """Helper function for interatively selecting an image to test out modifications from a stack of images

    Inputs
        stack: a 4D stack of images [slices, rows, cols, channels]
        slice_index: index of sliced image to use
        chan_index: index of the channel to use

    Returns
        slice_index: the slice index that was selected by the user
        chan_index: the chan_index that was selected by the user"""
    slice_name = stack.points.values[slice_index]
    img = stack.loc[slice_name, :, :, chan_name]
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img, cmap=mpl.cm.gray)
    return slice_name, chan_name


