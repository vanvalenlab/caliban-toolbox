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

from dcde.utils.io_utils import get_img_names

def choose_img(name, dirpath):
    filepath = os.path.join(dirpath, name)
    img = imread(filepath)
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img, cmap = mpl.cm.gray)
    return filepath

def choose_img_pair(frame, raw_dir, overlay_dir):
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
    

def edit_image(image, blur=1.0, sobel_toggle = True, sobel_factor = 100, invert_img = True, gamma_adjust = 1.0, equalize_hist=False, equalize_adapthist=False):
    """Used to edit the image using the widget tester"""

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
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(new_image, cmap = mpl.cm.gray)

    return new_image

def overlay_images(raw_img, overlay_img, prop_raw, v_min = 0, v_max = 255):
    
    prop_overlay = 1.0 - prop_raw
    mod_img = prop_overlay * overlay_img + prop_raw * raw_img
    #mod_img is currently a float, but will cause errors because values are not between 0 and 1
    mod_img = mod_img.astype(np.uint8)
    
    mod_img = sk.exposure.equalize_adapthist(mod_img, kernel_size=None, clip_limit=0.01, nbins=256)
    
    #equalize_adapthist outputs float64 image
    #rescale image to (0,255) before changing to uint8 dtype
    mod_img = sk.exposure.rescale_intensity(mod_img, in_range = "image", out_range = np.uint8)
    mod_img = mod_img.astype(np.uint8)
    
    #adjust brightness settings using v_min and v_max in range of uint8
    mod_img = sk.exposure.rescale_intensity(mod_img, in_range=(v_min, v_max))    
    
    #show modified image
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(mod_img, cmap = mpl.cm.gray)
    


