# Copyright 2016-2019 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-toolbox/LICENSE
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
Code for adjusting the contrast of images to aid image annotators
'''

import datetime
import json
import numpy as np
import os
import skimage as sk
import scipy
import stat
import sys

from deepcell_toolbox.utils.io_utils import get_image, get_img_names
from imageio import imread, imwrite
from scipy import ndimage
from skimage import filters


def contrast(image, sigma, hist, adapthist, gamma, sobel_option, sobel, invert, v_min, v_max):
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

    # Blur
    image = filters.gaussian(image, sigma, multichannel=False)

    # Find edges
    if sobel_option:
        image = sk.exposure.rescale_intensity(image, in_range = 'image', out_range = 'float')
        image += sobel * sk.filters.sobel(image)

    # Adjust gamma
    image = sk.exposure.adjust_gamma(image, gamma, gain = 1)

    # Invert
    if invert:
        image = sk.util.invert(image)

    if(hist):
        image = sk.exposure.equalize_hist(image, nbins=256, mask=None)

    if(adapthist):
        image = sk.exposure.rescale_intensity(image, in_range = 'image', out_range = 'float')
        image = sk.exposure.equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256)


    # Rescale intensity
    image = sk.exposure.rescale_intensity(image, in_range = 'image', out_range = np.uint8)
    image = image.astype(np.uint8)
    #okay to lose precision in these images--they don't get used in training data, just annotation

    image = sk.exposure.rescale_intensity(image, in_range=(v_min, v_max))    

    return image

def adjust_folder(base_dir, raw_folder, identifier, contrast_settings, is_2D):
    '''
    Constrast adjust images in a given folder, save contrast adjusted images in new folder. Also creates a
        json log to record settings used.
    
    Args:
        base_dir: full path to parent directory that holds raw_folder; json logs folder will be created here
        raw_folder: name of folder (not full path) containing images to be contrast adjusted
        identifier: string, used to name processed images and json log
        contrast_settings: dictionary of settings to use for contrast adjustment
        is_2D: whether to save images with 2D naming convention 
    
    Returns:
        None

    '''

    #directory specification, creating dirs when needed
    raw_dir = os.path.join(base_dir, raw_folder)

    #where will we save the processed files
    process_folder = raw_folder + "_contrast_adjusted"
    process_dir = os.path.join(base_dir, process_folder)
    if not os.path.isdir(process_dir):
        os.makedirs(process_dir)
        #add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(process_dir, mode)
    
    #where we will save a log of the settings
    log_dir = os.path.join(base_dir, "json_logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        #add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(log_dir, mode)

    #extract variables from settings dictionary
    sigma = contrast_settings['blur']
    hist = contrast_settings['equalize_hist']
    adapthist = contrast_settings['equalize_adapthist']
    gamma = contrast_settings['gamma_adjust']
    sobel_option = contrast_settings['sobel_toggle']
    sobel = contrast_settings['sobel_factor']
    invert = contrast_settings['invert_img']
    v_min = contrast_settings['v_min']
    v_max = contrast_settings['v_max']

    # Sorted list of image names from raw directory
    img_list = get_img_names(raw_dir)

    number_of_images = len(img_list)

    #Adjust contrast

    for j in range(number_of_images):

        img_path = os.path.join(raw_dir, img_list[j])
        image = get_image(img_path) #np.float32
        adjusted_image = contrast(image, sigma, hist, adapthist, gamma, sobel_option, sobel, invert, v_min, v_max)

        #Save processed image
        if is_2D:
            adjusted_name = os.path.join(process_dir, identifier + "_adjusted_img_" + str(j).zfill(3) + ".png")
        else:
            adjusted_name = os.path.join(process_dir, identifier + "_adjusted_frame_" + str(j).zfill(3) + ".png")
            
        imwrite(adjusted_name, adjusted_image)
        print("Saved " + adjusted_name + "; image " + str(j + 1) + " of " + str(number_of_images))
    
    print('Adjusted images have been saved in folder: ' + process_dir )

    #log in json for future reference

    log_data = {}
    log_data['date'] = str(datetime.datetime.now())
    log_data['raw_settings'] = contrast_settings
    log_data['raw'] = raw_folder
    log_data['identifier'] = identifier
    log_data['combined'] = False

    #save log in JSON format
    #save with identifier; should be saved in "log" folder

    log_path = os.path.join(log_dir, identifier + "_contrast_adjustment_log.json")

    with open(log_path, "w") as write_file:
        json.dump(log_data, write_file)

    print('A record of the settings used has been saved in folder: ' + log_dir)
    
    return None

def adjust_overlay(base_dir, raw_folder, overlay_folder, identifier, raw_settings, overlay_settings, combined_settings, is_2D):
    '''
    Constrast adjust images from two folders, overlay images, and save adjusted images in new folder. Also creates a
        json log to record settings used.
    
    Args:
        base_dir: full path to parent directory that holds raw_folder; json logs folder will be created here
        raw_folder: name of first folder (not full path) containing images to be contrast adjusted
        overlay_folder: name of second folder (not full path) containing images to be contrast adjusted
        identifier: string, used to name processed images and json log
        raw_settings: dictionary of settings to use for contrast adjustment of first folder
        overlay_settings: dictionary of settings to use for contrast adjustment of second folder
        combined_settings: dictionary of settings to use to overlay two images
        is_2D: whether to save images with 2D naming convention 
    
    Returns:
        None

    '''



    #directory management

    save_folder = raw_folder + "_overlay_" + overlay_folder

    save_dir = os.path.join(base_dir, save_folder)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        #add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(save_dir, mode)

    log_dir = os.path.join(base_dir, "json_logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        #add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(log_dir, mode)

    raw_dir = os.path.join(base_dir, raw_folder)
    overlay_dir = os.path.join(base_dir, overlay_folder)

    #extract variables from settings dictionaries

    raw_sigma = raw_settings['blur']
    raw_eq_adapthist = raw_settings['equalize_adapthist']
    raw_eq_hist = raw_settings['equalize_hist']
    raw_gamma = raw_settings['gamma_adjust']
    raw_invert = raw_settings['invert_img']
    raw_sobel_factor = raw_settings['sobel_factor']
    raw_sobel_toggle = raw_settings['sobel_toggle']
    raw_min = raw_settings['v_min']
    raw_max = raw_settings['v_max']

    overlay_sigma = overlay_settings['blur']
    overlay_eq_adapthist = overlay_settings['equalize_adapthist']
    overlay_eq_hist = overlay_settings['equalize_hist']
    overlay_gamma = overlay_settings['gamma_adjust']
    overlay_invert = overlay_settings['invert_img']
    overlay_sobel_factor = overlay_settings['sobel_factor']
    overlay_sobel_toggle = overlay_settings['sobel_toggle']
    overlay_min = overlay_settings['v_min']
    overlay_max = overlay_settings['v_max']

    prop_raw = combined_settings['prop_raw']
    v_min = combined_settings['v_min']
    v_max = combined_settings['v_max']

    #go image by image through dataset

    img_list = get_img_names(raw_dir)
    for frame in range(len(img_list)):

        #contrast adjust raw

        raw_img_name = get_img_names(raw_dir)[frame]
        raw_img_path = os.path.join(raw_dir, raw_img_name)
        raw_img = imread(raw_img_path)
        raw_adjusted = contrast(raw_img, raw_sigma, raw_eq_hist, raw_eq_adapthist, raw_gamma, raw_sobel_toggle, raw_sobel_factor, raw_invert, raw_min, raw_max)

        #contrast adjust overlay

        overlay_img_name = get_img_names(overlay_dir)[frame]
        overlay_img_path = os.path.join(overlay_dir, overlay_img_name)
        overlay_img = imread(overlay_img_path)
        overlay_adjusted = contrast(overlay_img, overlay_sigma, overlay_eq_hist, overlay_eq_adapthist, overlay_gamma, overlay_sobel_toggle, overlay_sobel_factor, overlay_invert, overlay_min, overlay_max)

        #combine images

        prop_overlay = 1.0 - prop_raw
        mod_img = overlay_adjusted * prop_overlay + raw_adjusted * prop_raw
        mod_img = sk.exposure.rescale_intensity(mod_img, in_range = 'image', out_range = 'uint8')
        mod_img = mod_img.astype(np.uint8)
        mod_img = sk.exposure.equalize_adapthist(mod_img, kernel_size=None, clip_limit=0.01, nbins=256)

        #equalize_adapthist outputs float64 image
        #rescale image to (0,255) before changing to uint8 dtype
        mod_img = sk.exposure.rescale_intensity(mod_img, in_range = "image", out_range = np.uint8)
        mod_img = mod_img.astype(np.uint8)

        #rescale brightness to user-defined range
        mod_img = sk.exposure.rescale_intensity(mod_img, in_range=(v_min, v_max))

        #name file
        if is_2D:
            adjusted_name = identifier + "_" + raw_folder + "_overlay_" + overlay_folder + "_img_" + str(frame).zfill(3) + ".png"
        else:
            adjusted_name = identifier + "_" + raw_folder + "_overlay_" + overlay_folder + "_frame_" + str(frame).zfill(3) + ".png"
        
        adjusted_img_path = os.path.join(save_dir, adjusted_name)

        #save image in new folder

        imwrite(adjusted_img_path, mod_img)
        print("Saved " + adjusted_name + "; image " + str(frame + 1) + " of " + str(len(img_list)))

    print("Adjusted images have been saved in folder: " + save_folder)

    #log in json for future reference

    log_data = {}
    log_data['date'] = str(datetime.datetime.now())
    log_data['raw_settings'] = raw_settings
    log_data['overlay_settings'] = overlay_settings
    log_data['combined_settings'] = combined_settings
    log_data['identifier'] = identifier
    log_data['overlay'] = overlay_folder
    log_data['raw'] = raw_folder
    log_data['combined'] = True

    #save log in JSON format
    #save with identifier; should be saved in "log" folder

    log_path = os.path.join(log_dir, identifier + "_contrast_adjustment_overlay_log.json")

    with open(log_path, "w") as write_file:
        json.dump(log_data, write_file)

    print('A record of the settings used has been saved in folder: ' + log_dir)
    
    return None

