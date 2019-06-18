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
Scripts for cleaning up figure eight annotations before the montage annotations are sliced into single frames;
clean_montage and relabel_montage are borrowed/adapted from deprecated scripts
'''

import numpy as np
import os
import warnings

from skimage.morphology import remove_small_holes, remove_small_objects
from imageio import imread, imwrite

from deepcell_toolbox.utils.io_utils import get_img_names


def grayscale_montage(image_path):
    '''
    Overwrite an image file with just the values in the first (red) channel; converts annotations downloaded
    from Figure 8 into single-channel masks

    Args:
        image_path: full path to image to convert to greyscale
        
    Returns:
        None
    '''
    image = imread(image_path)

    gray_image = np.zeros(image.shape[:2], dtype = "uint32") #creates 2D array for grayscale image
    gray_image = image[:,:,0] #uses values from red channel of rgb image to set intensities for 2D image

    imwrite(image_path, gray_image) #overwrites original .png
    
    return None

def convert_grayscale_all(annotations_folder):
    '''
    Convert all images in folder to greyscale
    
    Args:
        annotations_folder: full path to directory containing images to be converted to greyscale
        
    Returns:
        None
    '''

    #from folder, sort nicely, put images into list
    img_list = get_img_names(annotations_folder)

    for img in img_list:

        #load image
        img_name = os.path.join(annotations_folder, img)
        grayscale_montage(img_name)
        
    return None


def clean_montage(img_path):
    '''
    Remove small holes and small objects (stray pixels) from annotation. Original annotation will be
    overwritten by cleaned annotation
    
    Args:
        img_path: full path to image to clean
        
    Returns:
        None
    '''
    img = imread(img_path)

    clean_img = remove_small_holes(img, connectivity=1,in_place=False)
    clean_img = remove_small_objects(clean_img, min_size=10, connectivity=1, in_place=False)

    #empty array for new image
    fixed_img = np.zeros(img.shape, dtype= np.uint8)

    # using binary clean_img, assign cell labels to pixels of new image
    # iterate through the entire image
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # if clean_img is true, that pixel should have a label in fixed_img
            if clean_img[x, y] == True:
                #there should be a pixel here

                # get possible labels from original image (exclude background)
                ball = img[x-1:x+1, y-1:y+1].flatten()
                ball = np.delete(ball, np.where(ball == 0))

                if len(ball) == 0: # if no possible labels
                    if x>1 and y>1:
                        # take a larger area for labels, if possible
                        ball = img[x-2:x+2, y-2:y+2].flatten()
                        ball = np.delete(ball,np.where(ball == 0))

                        if len(ball) != 0:
                            # if there are possible values, take the most common
                            pixel_val = np.argmax(np.bincount(ball)).astype(np.uint8)
                            fixed_img[x, y] = pixel_val

                        else:
                            # otherwise take the label of that pixel from the original img
                            #   output location & frame to for user reference
                            fixed_img[x,y] = img[x,y]

                    else:
                        # otherwise take the label of that pixel from the original img
                        #   output location & frame to for user reference
                        fixed_img[x,y] = img[x,y]

                else: # if there are possible values, take the most common
                    pixel_val = np.argmax(np.bincount(ball)).astype(np.uint8)
                    fixed_img[x, y] = pixel_val

    imwrite(img_path, fixed_img)
    
    return None

def relabel_montage(y):
    '''
    Relabel annotations in an image, starting from 1, so that no labels are skipped
    
    Args:
        y: image to relabel, as numpy array
        
    Returns:
        relabeled image, as numpy array
    '''
    # create new_y to save new labels to
    new_y = np.zeros(y.shape, dtype = np.uint8)
    unique_cells = np.unique(y) # get all unique values of y
    unique_cells = np.delete(unique_cells, np.where(unique_cells == 0)) # remove 0, as it is background
    relabel_ids = np.arange(1, len(unique_cells) + 1)

    # iterate through existing labels, and save those pixels as new id in new_y
    for cell_id, relabel_id in zip(unique_cells, relabel_ids):
        cell_loc = np.where(y == cell_id)
        new_y[cell_loc] = relabel_id

    return new_y

def clean_montages(annotations_folder):
    '''
    Clean all annotations in folder
    
    Args:
        annotations_folder: full path to directory containing images to be cleaned
    
    Returns:
        None
    '''

    #from folder, sort nicely, put images into list
    img_list = get_img_names(annotations_folder)

    for img in img_list:

        #load image
        img_name = os.path.join(annotations_folder, img)

        #pass image to clean_montage
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean_montage(img_name)

        #save image
        #imwrite(img_name, cleaned_montage)

    return None

def relabel_montages(annotations_folder):
    '''
    Relabel all annotations in folder
    
    Args:
        annotations_folder: full path to directory containing images to be cleaned
    
    Returns:
        None
    '''

    #from folder, sort nicely, put images into list
    img_list = get_img_names(annotations_folder)

    for img in img_list:

        #load image
        img_name = os.path.join(annotations_folder, img)
        montage = imread(img_name, pilmode = 'I', as_gray = True)

        #pass image to relabel_montage
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            relabeled_montage = relabel_montage(montage)

        #save image
        imwrite(img_name, relabeled_montage)
        
    return None
    
