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
Code for adjusting the contrast of images to aid image annotaters
'''

from deepcell_toolbox.utils.io_utils import get_image, get_images_from_directory, get_img_names
import numpy as np
import skimage as sk
from skimage import filters
import os
from scipy import ndimage
import scipy
from imageio import imread, imwrite

def contrast(image, sigma, hist, adapthist, gamma, sobel_option, sobel, invert):
    '''takes image and image adjustment settings, returns adjusted image array'''

    if len(image.shape) > 2:
        print("Too many dimensions in your image. Make sure to split out channels and don't feed in image stacks")
        return

    # Blur
    image = filters.gaussian(image, sigma, multichannel=False)
    #nuclear_image = ndimage.filters.gaussian_filter(nuclear_image, sigma)

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

    return image

def adjust_folder(base_dir, raw_folder, identifier, sigma, hist, adapthist, gamma, sobel_option, sobel, invert):
    '''
    adjusts the contrast of raw images - does not overwrite raw images
    adjusted images are easier to crowdsource annotations

    assumes channels have already been split out of images for simplicity
    also because future versions will allow user to adjust variables for image processing, and different channels may need to be adjusted differently

    base_dir is parent folder that contains wherever raw data is stored, a folder to store processed images will be created in base_dir
    raw_folder is the name of the folder in base_dir where data to be contrast-adjusted is stored
    identifier used to name processed images

    '''

    #directory specification, creating dirs when needed
    raw_dir = os.path.join(base_dir, raw_folder)
    if not os.path.isdir(raw_dir):
        os.makedirs(raw_dir)
    #where will we save the processed files
    process_folder = raw_folder + "_contrast_adjusted"
    process_dir = os.path.join(base_dir, process_folder)
    if not os.path.isdir(process_dir):
        os.makedirs(process_dir)

    # Sorted list of image names from raw directory
    img_list = get_img_names(raw_dir)

    number_of_images = len(img_list)

    #Adjust contrast

    print('Processed data will be located at ' + process_dir )

    for j in range(number_of_images):

        print ( 'Processing image ' + str(j+1) + ' of ' + str(number_of_images))

        img_path = os.path.join(raw_dir, img_list[j])
        image = get_image(img_path) #np.float32
        adjusted_image = contrast(image, sigma, hist, adapthist, gamma, sobel_option, sobel, invert)

        #Save processed image

        nuclear_name = os.path.join(process_dir, identifier + "_adjusted_" + str(j).zfill(3) + '.png')
        imwrite(nuclear_name, adjusted_image)


def adjust_overlay(base_dir, raw_folder, overlay_folder, identifier, raw_settings, overlay_settings, combined_settings):

    #directory management

    save_folder = raw_folder + "_overlay_" + overlay_folder
    save_dir = os.path.join(base_dir, save_folder)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
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

    overlay_sigma = overlay_settings['blur']
    overlay_eq_adapthist = overlay_settings['equalize_adapthist']
    overlay_eq_hist = overlay_settings['equalize_hist']
    overlay_gamma = overlay_settings['gamma_adjust']
    overlay_invert = overlay_settings['invert_img']
    overlay_sobel_factor = overlay_settings['sobel_factor']
    overlay_sobel_toggle = overlay_settings['sobel_toggle']

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
        raw_adjusted = contrast(raw_img, raw_sigma, raw_eq_hist, raw_eq_adapthist, raw_gamma, raw_sobel_toggle, raw_sobel_factor, raw_invert)

        #contrast adjust overlay

        overlay_img_name = get_img_names(overlay_dir)[frame]
        overlay_img_path = os.path.join(overlay_dir, overlay_img_name)
        overlay_img = imread(overlay_img_path)
        overlay_adjusted = contrast(overlay_img, overlay_sigma, overlay_eq_hist, overlay_eq_adapthist, overlay_gamma, overlay_sobel_toggle, overlay_sobel_factor, overlay_invert)

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

        adjusted_img_name = identifier + "_" + raw_folder + "_overlay_" + overlay_folder + "_" + str(frame).zfill(3) + ".png"
        adjusted_img_path = os.path.join(save_dir, adjusted_img_name)

        #save image in new folder

        imwrite(adjusted_img_path, mod_img)
        print("Saved " + adjusted_img_name + "; image " + str(frame) + " of " + str(len(img_list)))
