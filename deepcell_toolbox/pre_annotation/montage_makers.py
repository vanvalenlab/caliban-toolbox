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

import os
import json
import math
import datetime

import numpy as np

from skimage.io import imread, imsave
from skimage.external import tifffile

from deepcell_toolbox.utils.io_utils import get_img_names


def montage_maker(montage_len, stack_direc, save_direc, identifier, x_pos, y_pos, row_length, x_buffer, y_buffer):
    '''
    montage_len = integer number of frames you want to be in the montage
    stack_direc = string, path to folder containing cropped images that will be turned into montage
    save_direc = string, path to folder where montages will be saved. usually .../montages
    identifier = string of information about the movie to be used in saving the montage. ie "RAW264_set1"
    x_pos = x coordinate of slice from original movie. eg 0 for the first column of slices
    y_pos = y coordinate of slice from original movie. eg 1 for the second row of slices
    row_length = integer number of frames you want per row of the montage
    x_buffer = how many pixels separating each column of images
    y_buffer = how many pixels separating each row of images
    '''

    #from folder, sort nicely, put images into list
    img_stack = get_img_names(stack_direc)

    #load test image
    test_img_name = os.path.join(stack_direc, img_stack[0])
    test_img = imread(test_img_name)

    #determine how many montages can be made from movie
    num_frames = len(img_stack)
    num_montages = num_frames // montage_len
    print("You will be able to make " + str(num_montages) + " montages from this movie.")
    print("The last %d frames will not be used in a montage. \n" % (num_frames % montage_len))

    ###check with user to confirm discarding last few images from movie if not divisible

    ###include option to make montage with remainder images? n by default

    #read image size and calculate montage size
    #test_img_name = cropped_images[0]
    #test_img = imread(os.path.join(raw_img_folder, test_img_name))
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
        montage_name = identifier + "_x_" + str(x_pos) + "_y_" + str(y_pos) + "_montage_" + str(montage) + ".png"
        montage_name = os.path.join(save_direc, montage_name)

        #loop through rows to add images to montage
        for row in range(number_of_rows):

            #set pixel range for current row
            y_start = y_buffer + ((y_buffer + y_dim) * row)
            y_end = (y_buffer + y_dim) * (row + 1)

            #loop through columns to add images to montage
            for column in range(row_length):

                #read img
                #this works because the img_stack is sorted_nicely
                slice_num = (montage * montage_len) + (row * row_length) + column
                current_slice_name = img_stack[slice_num]
                current_slice = imread(os.path.join(stack_direc, current_slice_name))

                #set pixel range for current column
                x_start = x_buffer + ((x_buffer + x_dim) * column )
                x_end = (x_buffer + x_dim) * (column + 1)

                #add image to montage
                montage_img[y_start:y_end,x_start:x_end] = current_slice

        #save montage
        imsave(montage_name, montage_img)

    return num_montages


def multiple_montage_maker(montage_len, direc, save_direc, identifier, num_x_segments, num_y_segments, row_length, x_buffer, y_buffer, log_direc):
    '''
    montage_len = integer number of frames you want to be in the montage
    direc = string, path to folder containing subfolders, each containing a cropped movie
    save_direc = string, path to folder where montages will be saved. usually .../montages
    identifier = string of information about the movie to be used in saving the montage. ie "RAW264_set1"
    num_x_segments = integer number of columns original movie was cropped into
    num_y_segments = integer number of rows original movie was cropped into
    row_length = integer number of frames you want per row of the montage
    x_buffer = how many pixels separating each column of images
    y_buffer = how many pixels separating each row of images
    '''

    #go to direc with cropped movie folders
    if not os.path.isdir(save_direc):
        os.makedirs(save_direc)

    #list of montages processed for logging purposes
    #montage_list = []

    #loop over cropped folders and call montage maker on each
    for y_pos in range(num_y_segments):

        for x_pos in range(num_x_segments):

            stack_direc = os.path.join(direc, identifier + "_x_" + str(x_pos).zfill(2) + "_y_" + str(y_pos).zfill(2))

            print("Now montaging images from: " + identifier + "_x_" + str(x_pos).zfill(2) + "_y_" + str(y_pos).zfill(2))
            num_montages = montage_maker(montage_len, stack_direc, save_direc, identifier, x_pos, y_pos, row_length, x_buffer, y_buffer)

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
    if not os.path.isdir(log_direc):
        os.makedirs(log_direc)
    log_path = os.path.join(log_direc, identifier + "_montage_maker_log.json")

    with open(log_path, "w") as write_file:
        json.dump(log_data, write_file)
