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

'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import json
import numpy as np
import os
import stat
import sys
import skimage as sk

from skimage.io import imsave
from skimage.external import tifffile
from deepcell_toolbox.utils.io_utils import get_image, get_img_names
from deepcell_toolbox.utils.misc_utils import sorted_nicely

def overlapping_img_chopper(img, save_dir, identifier, frame, num_x_segments, num_y_segments, overlap_perc, is_2D, file_ext):

    img_size = img.shape

    start_y = img_size[0]//num_y_segments
    overlapping_y_pix = int(start_y*(overlap_perc/100))
    new_y = int(start_y+2*overlapping_y_pix)

    start_x = img_size[1]//num_x_segments
    overlapping_x_pix = int(start_x*(overlap_perc/100))
    new_x = int(start_x+2*overlapping_x_pix)

    # pad full-size image
    padded_img = np.pad(img, ((overlapping_y_pix, overlapping_y_pix), (overlapping_x_pix, overlapping_x_pix)), mode='constant', constant_values=0)

    # make chopped images
    for i in range(num_x_segments):
        for j in range(num_y_segments):
            #take piece of full image and put in sub_img
            sub_img = padded_img[int(j*start_y):int(((j+1) * start_y) + (2 * overlapping_y_pix)),
                                             int(i*start_x):int(((i+1) * start_x) + (2 * overlapping_x_pix))]

            #save image, different naming conventions for 2D vs 3D
            if is_2D:
                sub_img_name = identifier + "_img_" + str(frame).zfill(3)+ "_x_" + str(i).zfill(2) + "_y_" + str(j).zfill(2) + file_ext

            else:
                sub_img_name = identifier + "_x_" + str(i).zfill(2) + "_y_" + str(j).zfill(2) + "_frame_" + str(frame).zfill(3) + file_ext
                
            #no need to save as .tif if original file isn't .tif,
            #and need to change intensities if saving as png    
            if file_ext == ".png":
                sub_img = sk.exposure.rescale_intensity(sub_img, in_range = 'image', out_range = np.uint8)
                sub_img = sub_img.astype(np.uint8)
            
            sub_img_path = os.path.join(save_dir, sub_img_name)
            imsave(sub_img_path, sub_img)


def overlapping_crop_dir(raw_direc, identifier, num_x_segments, num_y_segments, overlap_perc, frame_offset, is_2D = False):
    '''
    raw_direc = string, path to folder containing movie slices that will be cropped. likely ".../processed". passed to chopper
    num_x_segments = integer number of columns the movie will be chopped up into
    num_y_segments = integer number of rows the movie will be chopped up into
    overlap_perc = percent of image that will be added to border that overlaps with other chopped images
    '''
    #directories
    base_dir = os.path.dirname(raw_direc)
    unprocessed_name = os.path.basename(raw_direc)
    
    
    save_folder = unprocessed_name + "_offset_{0:03d}_chopped_{1:02d}_{2:02d}".format(frame_offset, num_x_segments, num_y_segments)
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

    # pick a file to calculate neccesary information from
    img_stack = get_img_names(raw_direc)

    #load test image
    test_img_name = os.path.join(raw_direc, img_stack[0])
    file_ext = os.path.splitext(img_stack[0])[1]

    test_img_temp = get_image(test_img_name)
    test_img_size = test_img_temp.shape
    img_squeeze = False

    print("Current Image Size: ", test_img_size)
    while True:
        start_flag = str(input("Correct dimensionality? (y/n): "))
        if start_flag != "y" and start_flag != "n":
            print("Please type y for 'yes' or n for 'no'")
            continue
        elif start_flag == "n":
            print("Making input 2D")
            test_img = np.squeeze(test_img_temp, axis=0)
            test_img_size = test_img.shape
            img_squeeze = True
            print("New Image Size: ", test_img.shape)
            break
        else:
            test_img = test_img_temp
            break   # input correct end loop

    # determine number of pixels required to achieve correct overlap
    start_y = test_img_size[0]//num_y_segments
    overlapping_y_pix = int(start_y*(overlap_perc/100))
    new_y = int(start_y+2*overlapping_y_pix)

    start_x = test_img_size[1]//num_x_segments
    overlapping_x_pix = int(start_x*(overlap_perc/100))
    new_x = int(start_x+2*overlapping_x_pix)

    #for images that don't aren't divided evenly by num_segments, restitching can lead to losing a few pixels
    #avoid this by logging the actual start indices for the subimages
    y_start_indices = []
    x_start_indices = []

    for i in range(num_x_segments):
        x_start_index = int(i*start_x) 
        x_start_indices.append(x_start_index)

    for j in range(num_y_segments):
        y_start_index = int(j*start_y)
        y_start_indices.append(y_start_index)

    print("Your new images will be ", new_x, " pixels by ", new_y, " pixels in size.")

    print("Processing...")

    files = os.listdir(raw_direc)
    files_sorted = sorted_nicely(files)

    for frame, file in enumerate(files_sorted):
        # load image
        file_path = os.path.join(raw_direc, file)
        if img_squeeze == False:
            img = get_image(file_path)
        else:
            img = np.squeeze(get_image(file_path), axis=0)
        #factor in whether we're starting from frame zero so that chopped files get correct frame number
        current_frame = frame + frame_offset

        #each frame of the movie will be chopped into x by y smaller frames and saved
        overlapping_img_chopper(img, save_dir, identifier, current_frame, num_x_segments, num_y_segments, overlap_perc, is_2D, file_ext)

    #log in json for post-annotation use

    log_data = {}
    log_data['date'] = str(datetime.datetime.now())
    log_data['num_x_segments'] = num_x_segments
    log_data['num_y_segments'] = num_y_segments
    log_data['overlap_perc'] = overlap_perc
    log_data['identifier'] = identifier
    log_data['y_start_indices'] = y_start_indices
    log_data['x_start_indices'] = x_start_indices
    log_data['overlapping_x_pix'] = overlapping_x_pix
    log_data['overlapping_y_pix'] = overlapping_y_pix
    log_data['original_y'] = test_img_size[0]
    log_data['original_x'] = test_img_size[1]
    log_data['frame_offset'] = frame_offset
    log_data['num_images'] = len(files_sorted)


    #save log in JSON format
    #save with identifier; should be saved in "log" folder

    log_path = os.path.join(log_dir, identifier + "_overlapping_chopper_log.json")

    with open(log_path, "w") as write_file:
        json.dump(log_data, write_file)

    print("Cropped files saved to {}".format(save_dir))
