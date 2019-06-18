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
Functions for processing montage annotations into individual frames of a movie
'''

from __future__ import division

import json
import math
import numpy as np
import os
import stat
import sys
import warnings

#from skimage.io import imsave
from imageio import imread, imwrite
from deepcell_toolbox.utils.io_utils import get_img_names



def read_json_params_montage(log_folder, identifier):
    '''
    Reads parameters from a json log created by the montage maker. Not currently used in pipeline.
    
    Args:
        log_folder: full path to folder where json logs are stored
        identifier: string used to specify data set (same variable used throughout pipeline); 
            used to load correct json file

    Returns:
        Variables extracted from dictionary; montage_len, num_x_segments, num_y_segments, row_length, 
            x_buffer, y_buffer, num_montages
    '''

    json_path = os.path.join(log_folder, identifier + "_montage_maker_log.json")

    with open(json_path) as json_file:
        log_data = json.load(json_file)

        montage_len = log_data['montage_len']
        num_x_segments = log_data['num_x_segments']
        num_y_segments = log_data['num_y_segments']
        row_length = log_data['row_length']
        x_buffer = log_data['x_buffer']
        y_buffer = log_data['y_buffer']
        num_montages = log_data['montages_in_pos']

    return montage_len, num_x_segments, num_y_segments, row_length, x_buffer, y_buffer, num_montages

def read_json_params_chopper(log_folder, identifier):
    '''
    Reads parameters from a json log created by the overlapping chopper. Not currently used in pipeline.
    
    Args:
        log_folder: full path to folder where json logs are stored
        identifier: string used to specify data set (same variable used throughout pipeline); 
            used to load correct json file

    Returns:
        Variables extracted from dictionary; overlap_perc, num_x_segments, num_y_segments
    '''


    json_path = os.path.join(log_folder, identifier + "_overlapping_chopper_log.json")

    with open(json_path) as json_file:
        log_data = json.load(json_file)

        overlap_perc = log_data['overlap_perc']
        num_x_segments = log_data['num_x_segments']
        num_y_segments = log_data['num_y_segments']

    return overlap_perc, num_x_segments, num_y_segments



def montage_chopper(montage_path, identifier, montage_len, part_num, x_seg, y_seg, row_length, x_buffer, y_buffer, save_dir):
    '''
    Takes a montage and saves its constituent frames as individual image files
    
    Args:
        montage_path: full path to montage annotation that will be chopped into constituent frames
        identifier: string used to specify data set (same variable used throughout pipeline); 
            used to save image pieces
        montage_len: how many frames in total are in montage
        part_num: which montage number is being chopped (eg, montage 2 out of 10 for position x, y)
        x_seg: which x position is being chopped
        y_seg: which y position is being chopped
        row_length: number of frames in each row of montage
        x_buffer: how many pixels separate each column of images
        y_buffer: how many pixels separate each row of images
        save_dir: full path to directory where individual frames will be saved
        
    Returns:
        None
    
    '''

    #montage_num is to calculate correct frame number for new image: ie, montage_num*montage_len is the starting index for frame numbering

    num_rows = math.ceil(montage_len/row_length)

    #load montage
    montage_img = imread(montage_path)
    #assumes you're passing in a .png montage
    montage_x = montage_img.shape[1]
    montage_y = montage_img.shape[0]

    #dimensions for each frame
    x_dim = (montage_x - ((row_length + 1) * x_buffer))//row_length
    y_dim = (montage_y - ((num_rows + 1) * y_buffer))//num_rows

    for row in range(num_rows):
        #math to calculate pixel boundaries, y
        y_start = y_buffer + ((y_buffer + y_dim) * row)
        y_end = (y_buffer + y_dim) * (row + 1)

        for column in range(row_length):

            #math to calculate frame number; each movie from a montage will start at frame zero
            frame_num = (row * row_length) + column

            #make image name
            #not anticipating more than 99 x 99 segments, or more than 99 frames per montage
            #if this changes, need to change zfill here for consistent naming
            current_frame_name = "{0}_x_{1:02d}_y_{2:02d}_part_{3}_frame_{4:02d}.png".format(identifier, x_seg, y_seg, part_num, frame_num)
            current_frame_path = os.path.join(save_dir, current_frame_name)

            #math to calculate pixel boundaries, x
            x_start = x_buffer + ((x_buffer + x_dim) * column )
            x_end = (x_buffer + x_dim) * (column + 1)

            #make np.array to hold image info
            current_frame = np.zeros((y_dim, x_dim), dtype = np.uint16)

            #copy selected area of montage into the np.array
            current_frame = montage_img[y_start:y_end,x_start:x_end]

            #save image
            imwrite(current_frame_path, current_frame)
    
    return None

def all_montages_chopper(base_dir, montage_dir, identifier, json_montage_log):
    '''
    Runs montage_chopper on all montages in given folder
    
    Args:
        base_dir: full path to directory where folder to keep image pieces will be created
        montage_dir: full path to directory where montage annotations are saved
        identifier: string used to specify data set (same variable used throughout pipeline); 
            used to load montages and save image pieces
        json_montage_log: dictionary of variables loaded from json log created by montage_maker
        
    Returns:
        None
    '''
    
    #unpack info from json log
    montage_len = json_montage_log['montage_len']
    num_x_segments = json_montage_log['num_x_segments']
    num_y_segments = json_montage_log['num_y_segments']
    row_length = json_montage_log['row_length']
    x_buffer = json_montage_log['x_buffer']
    y_buffer = json_montage_log['y_buffer']
    num_montages = json_montage_log['montages_in_pos']

    save_dir = os.path.join(base_dir, "chopped_annotations")
    perm_mod = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        os.chmod(save_dir, perm_mod)
    
    for part_num in range(num_montages):
        for x_seg in range(num_x_segments):
            for y_seg in range(num_y_segments):

                #all montages for that position should get chopped into that folder
                montage_name = "{0}_x_{1}_y_{2}_montage_{3}_annotation.png".format(identifier, x_seg, y_seg, part_num)
                montage_path = os.path.join(montage_dir, montage_name)

                if not os.path.isfile(montage_path):
                    print("Didn't find a file at: ", montage_path)
                else:
                    #run the montage chopper on a file that exists
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        montage_chopper(montage_path, identifier, montage_len, part_num, x_seg, y_seg, row_length, x_buffer, y_buffer, save_dir)


    return None


def overlapping_img_chopper(img_path, save_dir, identifier, frame, num_x_segments, num_y_segments, overlap_perc):
    '''
    Slightly modified version of pre-annotation overlapping chopper to be used with raw_movie_maker; not
    currently used in pipeline.
    
    Args:
        img_path: full path to image to be chopped
        save_dir: full path to directory where chopped image pieces will be saved
        identifier: string used to specify data set (same variable used throughout pipeline), used to save images
        frame: which frame image pieces come from, used to save images
        num_x_segments: number of columns image will be chopped into
        num_y_segments: number of rows image will be chopped into
        overlap_perc: percent of image on each edge that overlaps with other chopped images
        
    Returns:
        None
    '''

    img = imread(img_path)

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

            # save sub image
            sub_img_name = identifier + "_raw_x_" + str(i).zfill(2) + "_y_" + str(j).zfill(2) + "_frame_" + str(frame).zfill(3) + '.tif'
            subdir_name = "x_" + str(i).zfill(2) + '_y_' + str(j).zfill(2)
            sub_img_path = os.path.join(save_dir, subdir_name, "raw", sub_img_name)
            #import pdb; pdb.set_trace()
            imsave(sub_img_path, sub_img)
            
    return None


def raw_movie_maker(base_dir, raw_dir, identifier):
    '''
    Chops raw images into pieces to match the annotation pieces; not currently used in pipeline.
    
    Args:
        base_dir: full path to directory that contains folder for json logs; "movies" folder will be created
            in this directory
        raw_dir: full path to directory that contains raw images to be chopped
        identifier: string used to specify data set (same variable used throughout pipeline), used to load files
            and save images

    Returns:
        None
    '''

    movies_dir = os.path.join(base_dir, "movies")
    if not os.path.isdir(movies_dir):
        os.makedirs(movies_dir)
        #add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(movies_dir, mode)

    #find json data in folder log was saved to; "json_logs" by default
    json_folder = os.path.join(base_dir, "json_logs")

    #get data, store as tuples
    json_params_montage = read_json_params_montage(json_folder, identifier)

    montage_len = json_params_montage[0]
    num_x_segments_m = json_params_montage[1] #check to make sure these are getting chopped into same number of segments
    num_y_segments_m = json_params_montage[2]
    num_montages = json_params_montage[6]

    json_params_chopper = read_json_params_chopper(json_folder, identifier)

    overlap_perc = json_params_chopper[0]
    num_x_segments = json_params_chopper[1]
    num_y_segments = json_params_chopper[2]

    #sanity check
    if num_x_segments == num_x_segments_m and num_y_segments == num_y_segments_m:

        #make folders for raw
        for part in range(num_montages):
            part_folder = os.path.join(movies_dir, "part" + str(part)) #zero based indexing for part folders
            if not os.path.isdir(part_folder):
                os.makedirs(part_folder)
                #add folder modification permissions to deal with files from file explorer
                mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
                os.chmod(part_folder, mode)

            for x_seg in range(num_x_segments):
                for y_seg in range(num_y_segments):

                    #make folder for that position
                    position_folder = os.path.join(part_folder, "x_{0:02d}_y_{1:02d}".format(x_seg, y_seg))
                    raw_subfolder = os.path.join(position_folder, "raw")
                    if not os.path.isdir(raw_subfolder):
                        os.makedirs(raw_subfolder)
                        #add folder modification permissions to deal with files from file explorer
                        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
                        os.chmod(raw_subfolder, mode)


        #sorting of raw movies into appropriate parts folders happens here
        total_frames = num_montages * montage_len
        raw_img_list = get_img_names(raw_dir) #get_img_names sorts them

        #don't chop more raw frames than we have annotated frames
        for frame in range(total_frames):
            #frame numbers will start from zero in each part
            #if needed in future can put into one continuous movie with frame = relative_frame + (part_num * montage_len)
            current_part = frame//montage_len
            relative_frame = frame - (current_part * montage_len)

            img_file = raw_img_list[frame]
            img_path = os.path.join(raw_dir, img_file)

            save_dir = os.path.join(movies_dir, "part" + str(current_part))
            #sorting of chopped pieces into different position folders happens in the chopper
            #chop
            with warnings.catch_warnings(): #ignore "low contrast image" warning
                warnings.simplefilter("ignore")
                overlapping_img_chopper(img_path, save_dir, identifier, relative_frame, num_x_segments, num_y_segments, overlap_perc)


        #frames in each part should start at zero; if they need to be stitched back together, scripts should utilize the "montage_len" variable to calculate offset

    else:
        print("Num_segments mismatch; double-check your files and logs to make sure you're trying to put the correct movies together.")
        
    return None
    
