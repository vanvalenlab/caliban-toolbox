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
Put consecutive frames of a movie next to each other for annotation jobs that track cells across frames
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import json
import math
import numpy as np
import os
import stat
import sys
import warnings

from deepcell_toolbox.utils.io_utils import get_img_names
from skimage.external import tifffile
from skimage.io import imread, imsave


def montage_maker(montage_len, chopped_dir, save_dir, identifier, x_pos, y_pos, row_length, x_buffer, y_buffer):
    '''
    Take a stack of images from a specific x and y position in a movie and create montages
    
    Args:
        montage_len: integer number of frames montage will contain
        chopped_dir: full path to folder containing cropped images that will be turned into montages
        save_dir: full path to folder where montages will be saved
        identifier: string used to specify data set (same variable used throughout pipeline); 
            used to load image pieces and to save montages
        x_pos: x coordinate of slice from original movie, used to load image pieces and to save montage
        y_pos: y coordinate of slice from original movie, used to load image pieces and to save montage
        row_length: integer number of frames each row of the montage will hold
        x_buffer: how many pixels will separate each column of images
        y_buffer: how many pixels will separate each row of images
        
    Returns:
        The number of montages created from the given image stack
    '''

    #from folder, sort nicely, put images into list
    img_stack = get_img_names(chopped_dir)

    #load test image
    test_img_name = os.path.join(chopped_dir, img_stack[0])
    test_img = imread(test_img_name)

    #get file_ext from test_img
    file_ext = os.path.splitext(test_img_name)[1]

    #determine how many montages can be made from movie
    subimg_list = [img for img in img_stack if "x_{0:02d}_y_{1:02d}".format(x_pos, y_pos) in img]
    num_frames = len(subimg_list)
    num_montages = num_frames // montage_len
    print("You will be able to make " + str(num_montages) + " montages from this movie.")
    print("The last %d frame(s) will not be used in a montage. \n" % (num_frames % montage_len))

    ###check with user to confirm discarding last few images from movie if not divisible

    ###include option to make montage with remainder images? n by default

    #read image size and calculate montage size
    x_dim = test_img.shape[-1]
    y_dim = test_img.shape[-2]

    number_of_rows = math.ceil(montage_len/row_length)

    final_x = (row_length * x_dim) + ((row_length + 1) * x_buffer)
    final_y = (number_of_rows * y_dim) + ((number_of_rows + 1) * y_buffer)

    #loop through num_montages to make more than one montage per input movie if possible
    for montage in range(num_montages):

        #make array to hold montage
        montage_img = np.zeros((final_y, final_x), dtype = np.uint16)

        #name the montage
        montage_name = identifier + "_x_" + str(x_pos).zfill(2) + "_y_" + str(y_pos).zfill(2) + "_montage_" + str(montage).zfill(2) + ".png"
        montage_name = os.path.join(save_dir, montage_name)

        #loop through rows to add images to montage
        for row in range(number_of_rows):

            #set pixel range for current row
            y_start = y_buffer + ((y_buffer + y_dim) * row)
            y_end = (y_buffer + y_dim) * (row + 1)

            #loop through columns to add images to montage
            for column in range(row_length):

                #read img
                #this works because the images were saved in 3D naming mode
                frame_num = (montage * montage_len) + (row * row_length) + column
                current_frame_name = "{0}_x_{1:02d}_y_{2:02d}_frame_{3:03d}{4}".format(identifier, x_pos, y_pos, frame_num, file_ext)
                current_slice = imread(os.path.join(chopped_dir, current_frame_name))

                #set pixel range for current column
                x_start = x_buffer + ((x_buffer + x_dim) * column )
                x_end = (x_buffer + x_dim) * (column + 1)

                #add image to montage
                montage_img[y_start:y_end,x_start:x_end] = current_slice

        #save montage
        with warnings.catch_warnings():
            #ignore "low contrast image" warnings
            warnings.simplefilter("ignore")
            imsave(montage_name, montage_img)

    return num_montages


def multiple_montage_maker(montage_len, base_dir, chopped_dir, save_dir, identifier, num_x_segments, num_y_segments, row_length, x_buffer, y_buffer):
    '''
    Create montages from all x and y positions of a chopped movie. Also saves json log of settings used, to be
    used later to get individual frames from montage.

    Args:
        montage_len: integer number of frames montage will contain
        base_dir: full path to parent directory that contains json logs folder
        chopped_dir: full path to folder containing cropped images that will be turned into montages
        save_dir: full path to folder where montages will be saved
        identifier: string used to specify data set (same variable used throughout pipeline); 
            used to load image pieces and to save montages 
        num_x_segments: how many pieces across the movie was chopped into
        num_y_segments: how many pieces down the movie was chopped into
        row_length: integer number of frames each row of the montage will hold
        x_buffer: how many pixels will separate each column of images
        y_buffer: how many pixels will separate each row of images

    Returns:
        None
    '''

    #make folder to save montages in if it doesn't exist already
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        #add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(save_dir, mode)

    #make log_dir if it doesn't exist already
    log_dir = os.path.join(base_dir, "json_logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        #add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(log_dir, mode)

    #list of montages processed for logging purposes
    #montage_list = []

    #loop over positions of cropped images and call montage maker on each
    for y_pos in range(num_y_segments):
        for x_pos in range(num_x_segments):

            print("Now montaging images from the stack of: " + identifier + "_x_" + str(x_pos).zfill(2) + "_y_" + str(y_pos).zfill(2))
            num_montages = montage_maker(montage_len, chopped_dir, save_dir, identifier, x_pos, y_pos, row_length, x_buffer, y_buffer)

            #logged_name = identifier + "_x_" + str(x_pos) + "_y_" + str(y_pos)
            #montage_list.append(logged_name)

    #make log of relevant info
    log_data = {}
    log_data['date'] = str(datetime.datetime.now())
    #log_data['montages_made'] = montage_list
    log_data['montage_len'] = montage_len
    log_data['identifier'] = identifier
    log_data['num_x_segments'] = num_x_segments
    log_data['num_y_segments'] = num_y_segments
    log_data['row_length'] = row_length
    log_data['x_buffer'] = x_buffer
    log_data['y_buffer'] = y_buffer
    log_data['montages_in_pos'] = num_montages

    #save log in JSON format
    #save with identifier; should be saved in "log" folder
    log_path = os.path.join(log_dir, identifier + "_montage_maker_log.json")

    with open(log_path, "w") as write_file:
        json.dump(log_data, write_file)

    print('A record of the settings used has been saved in folder: ' + log_dir)
    
    return None

