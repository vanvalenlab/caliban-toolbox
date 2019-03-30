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

from __future__ import division

import json
import math
import numpy as np
import os
import warnings

from skimage.io import imsave
from imageio import imread
from deepcell_toolbox.utils.io_utils import get_img_names



def read_json_params_montage(log_folder, identifier):
    '''
    finds the .json file that you want to read montage parameters from
    returns the parameters you need
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
    finds the .json file that you want to read overlapping_chopper parameters from
    returns the parameters you need
    '''

    json_path = os.path.join(log_folder, identifier + "_overlapping_chopper_log.json")

    with open(json_path) as json_file:
        log_data = json.load(json_file)

        overlap_perc = log_data['overlap_perc']
        num_x_segments = log_data['num_x_segments']
        num_y_segments = log_data['num_y_segments']

    return overlap_perc, num_x_segments, num_y_segments



def montage_chopper(montage_path, identifier, montage_len, part_num, x_seg, y_seg, row_length, x_buffer, y_buffer, save_folder):
    '''
    takes the annotation of a single montage and chops it into pieces
    these pieces match up with the movies of contrast-adjusted raw images that were processed into montages
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
            #not anticipating more than 99 x 99 segments, or more than 999 frames
            #if this changes, need to change zfill here for consistent naming
            current_frame_name = identifier + "_x_" + str(x_seg).zfill(2) + "_y_" + str(y_seg).zfill(2) + "_frame_" + str(frame_num).zfill(3) + ".png"
            current_frame_path = os.path.join(save_folder, current_frame_name)

            #math to calculate pixel boundaries, x
            x_start = x_buffer + ((x_buffer + x_dim) * column )
            x_end = (x_buffer + x_dim) * (column + 1)

            #make np.array to hold image info
            current_frame = np.zeros((y_dim, x_dim), dtype = np.uint16)

            #copy selected area of montage into the np.array
            current_frame = montage_img[y_start:y_end,x_start:x_end]

            #save image
            imsave(current_frame_path, current_frame)

def all_montages_chopper(base_folder, identifier):
    '''
    calls read_json_params so it can pass correct information to montage_chopper
    calls montage_chopper on all of the montages in given folder
    does some organizing of save folders for montage_chopper
    '''

    #find json data in folder log was saved to; "json_logs" by default
    json_folder = os.path.join(base_folder, "json_logs")
    #get data, store as a tuple
    json_params = read_json_params_montage(json_folder, identifier)

    montage_len = json_params[0]
    num_x_segments = json_params[1]
    num_y_segments = json_params[2]
    row_length = json_params[3]
    x_buffer = json_params[4]
    y_buffer = json_params[5]
    num_montages = json_params[6]

    montage_folder = os.path.join(base_folder, "annotations") #where to find the montages

    movie_folder = os.path.join(base_folder, "movies")
    for part_num in range(num_montages):

        part = "part" + str(part_num)

        for x_seg in range(num_x_segments):
            for y_seg in range(num_y_segments):

                #make folder for that position
                position_folder = os.path.join(movie_folder, part, "x_{0:02d}_y_{1:02d}".format(x_seg, y_seg))
                annotations_folder = os.path.join(position_folder, "annotated") #where to save the frames
                if not os.path.isdir(annotations_folder):
                    os.makedirs(annotations_folder)

                #all montages for that position should get chopped into that folder

                montage_name = identifier + "_x_" + str(x_seg) + "_y_" + str(y_seg) + "_montage_" + str(part_num) + "_annotation.png"
                montage_path = os.path.join(montage_folder, montage_name)

                if not os.path.isfile(montage_path):
                    print("Didn't find a file at: ", montage_path)
                else:
                    #run the montage chopper on a file that exists
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        montage_chopper(montage_path, identifier, montage_len, part_num, x_seg, y_seg, row_length, x_buffer, y_buffer, annotations_folder)


    #name: MouseBrain_s7_nuc_x_0_y_2_montage_0_annotation
    #name: {identifier}_x_{}_y{}_montage_{}_annotation.png



def overlapping_img_chopper(img_path, save_dir, identifier, frame, num_x_segments, num_y_segments, overlap_perc):
    '''
    slightly modified from overlapping_chopper in pre-annotation notebooks, mostly to save files in different places
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


def raw_movie_maker(base_dir, raw_dir, identifier):
    '''
    base_dir is folder that contains folders for log files, movies, etc
    raw_dir is the name of the folder that contains the raw images that will be chopped up; often "raw" or identifier

    to chop up the raw images to match the annotations, script pulls from json log that was made when contrast adjusted images were cropped; this avoids user input to match chopped images with each other
    also pulls from montage_maker json log so that the correct number of raw frames are matched to annotations
    '''

    movies_dir = os.path.join(base_dir, "movies")
    if not os.path.isdir(movies_dir):
        os.makedirs(movies_dir)

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
            for x_seg in range(num_x_segments):
                for y_seg in range(num_y_segments):

                    #make folder for that position
                    position_folder = os.path.join(part_folder, "x_{0:02d}_y_{1:02d}".format(x_seg, y_seg))
                    raw_subfolder = os.path.join(position_folder, "raw")
                    if not os.path.isdir(raw_subfolder):
                        os.makedirs(raw_subfolder)

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
