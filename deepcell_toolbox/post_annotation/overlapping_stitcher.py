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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import numpy as np
import os
import sys
import stat

from imageio import imread, imwrite
from deepcell_toolbox.utils.io_utils import get_image, get_img_names

def overlapping_stitcher_core(pieces_dir, 
                                 save_dir, 
                                 x_start_indices, 
                                 y_start_indices, 
                                 overlapping_x_pix, 
                                 overlapping_y_pix, 
                                 original_x, 
                                 original_y, 
                                 identifier, 
                                 sub_img_format, 
                                 save_name):

    #save path
    img_path = os.path.join(save_dir, save_name)
    
    #load test image
    test_img_name = os.listdir(pieces_dir)[0]
    test_img_path = os.path.join(pieces_dir, test_img_name)
    test_img = get_image(test_img_path)
    
    #make empty image, includes zero padding around edges
    
    small_img_size = test_img.shape
    
    #trim dimensions are the size of each sub image, with the overlap removed
    trim_y = small_img_size[0] - 2*overlapping_y_pix
    trim_x = small_img_size[1] - 2*overlapping_x_pix
    
    #pad dimensions are the amount of padding added to a sub image on each edge
    pad_x = overlapping_x_pix
    pad_y = overlapping_y_pix
    
    full_y = original_y
    full_x = original_x
    
    padded_full_img = np.zeros((full_y + 2*pad_y, full_x + 2*pad_x))

    #loop through the image pieces (sub_imgs)
    for j in range(len(y_start_indices)):
        for i in range(len(x_start_indices)):

            #load sub_img
            sub_img_name = sub_img_format.format(identifier,i,j)
            sub_img_path = os.path.join(pieces_dir, sub_img_name)
            try:
                sub_img = get_image(sub_img_path)
            except:
                sub_img = np.zeros(small_img_size)
            
            sub_x = sub_img.shape[1]
            sub_y = sub_img.shape[0]
            
            x_start = x_start_indices[i]
            y_start = y_start_indices[j]
            
            #don't repeat values even though each sub image has similar values
            lowest_allowed_val = np.amax(padded_full_img)
            sub_img = np.where(sub_img == 0, sub_img, sub_img + lowest_allowed_val)
            
            #if non-zero values in the sub image and the full image overlap, replace all instances of
            #that value in the sub image, using most common value, keep other values in the sub image the same
            
            #working on top edge of sub_img
            sub_img_overlap = sub_img[0:2*pad_y, :]
            full_img_overlap = padded_full_img[y_start:y_start+2*pad_y, x_start:x_start+sub_x]
            
            #add in cells from full_image that overlap with sub_img background
            merged_overlap = np.where(sub_img_overlap == 0, full_img_overlap, sub_img_overlap)
            sub_img[0:2*pad_y, :] = merged_overlap
            
            #get ids of cells in sub_img that have any pixels in this overlap region
            sub_cells = np.unique(merged_overlap)
            nonzero_sub_cells = sub_cells[np.nonzero(sub_cells)]
            
            #for each cell, figure out which cell in full_img overlaps it the most
            for cell in nonzero_sub_cells:
                overlaps = np.where(merged_overlap == cell, full_img_overlap, 0)
                nonzero_overlaps = overlaps[np.nonzero(overlaps)]
                if len(nonzero_overlaps) > 0:
                    (values,counts) = np.unique(nonzero_overlaps,return_counts=True)
                    ind=np.argmax(counts)
                    overlapper = values[ind]
                    
                    #use the best overlapping cell to update the values in the whole sub_img
                    #not just the region of the sub_img that directly overlaps
                    
                    #replaces "cell" values with the overlapper, leaves everything else unchanged
                    sub_img = np.where(sub_img == cell, overlapper, sub_img)
            
            #working on left edge of sub_img
            sub_img_overlap = sub_img[:, 0:2*pad_x]
            full_img_overlap = padded_full_img[y_start:y_start+sub_y, x_start:x_start+2*pad_x]
            
            #add in cells from full_image that overlap with sub_img background
            merged_overlap = np.where(sub_img_overlap == 0, full_img_overlap, sub_img_overlap)
            sub_img[:, 0:2*pad_x] = merged_overlap
            
            #get ids of cells in sub_img that have any pixels in this overlap region
            sub_cells = np.unique(merged_overlap)
            nonzero_sub_cells = sub_cells[np.nonzero(sub_cells)]

            for cell in nonzero_sub_cells:
                overlaps = np.where(merged_overlap == cell, full_img_overlap, 0)
                nonzero_overlaps = overlaps[np.nonzero(overlaps)]
                if len(nonzero_overlaps) > 0:
                    (values,counts) = np.unique(nonzero_overlaps,return_counts=True)
                    ind=np.argmax(counts)
                    overlapper = values[ind]
                    
                    #use the best overlapping cell to update the values in the whole sub_img
                    #not just the region of the sub_img that directly overlaps
                    
                    #replaces "cell" values with the overlapper, leaves everything else unchanged
                    sub_img = np.where(sub_img == cell, overlapper, sub_img)
            
            #put sub image into the full image
            padded_full_img[y_start:y_start+sub_y, x_start:x_start+sub_x] = sub_img
            
    padded_full_img = padded_full_img.astype(np.uint16)

    #trim off edges
    full_img = padded_full_img[pad_y:pad_y+full_y, pad_x:pad_x+full_x]
    
    #relabel cells so values aren't skipped
    relabeled_img = np.zeros(full_img.shape, dtype = np.uint16)
    unique_cells = np.unique(full_img) # get all unique values of y
    unique_cells = unique_cells[np.nonzero(unique_cells)]
    relabel_ids = np.arange(1, len(unique_cells) + 1)

    for cell_id, relabel_id in zip(unique_cells, relabel_ids):
        relabeled_img = np.where(full_img == cell_id, relabel_id, relabeled_img)

    #save big image
    imwrite(img_path, relabeled_img)

def overlapping_stitcher_folder(pieces_dir, save_dir, identifier, num_images, json_chopper_log, is_2D):

    #directories
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        #add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(save_dir, mode)
        
    pieces_list = get_img_names(pieces_dir)
        
    #load test image for parameters if needed
    test_img = imread(os.path.join(pieces_dir, pieces_list[0]))
    small_img_size = test_img.shape
    
    #load variables from json log
    overlap_perc = json_chopper_log['overlap_perc']
    num_x_segments = json_chopper_log['num_x_segments']
    num_y_segments = json_chopper_log['num_y_segments']
    
    try:
        x_start_indices = json_chopper_log['x_start_indices']
        y_start_indices = json_chopper_log['y_start_indices']
        overlapping_x_pix = json_chopper_log['overlapping_x_pix']
        overlapping_y_pix = json_chopper_log['overlapping_y_pix']
        original_x = json_chopper_log['original_x']
        original_y = json_chopper_log['original_y']
        
    #preferable to load these variables 
    #but if they aren't in the log (outdated log or other reason?), calculate them    
    except:
    
        #trim dimensions are the size of each sub image, with the overlap removed
        trim_y = math.ceil(small_img_size[0]/(1+(2*overlap_perc/100)))
        trim_x = math.ceil(small_img_size[1]/(1+(2*overlap_perc/100)))
        
        x_start_indices = []
        for i in range(num_x_segments):
            x_start_index = int(i*trim_x) 
            x_start_indices.append(x_start_index)
            
        y_start_indices = []
        for j in range(num_y_segments):
            y_start_index = int(j*trim_y)
            y_start_indices.append(y_start_index)
        
        overlapping_x_pix = (small_img_size[1] - trim_x) //2
        overlapping_y_pix = (small_img_size[0] - trim_y) //2
        #print('overlapping pix x, y', overlapping_x_pix, overlapping_y_pix)
    
        #not sure if overlapping_pix or pad is better calculation so leaving this in
        #pad_x = int((overlap_perc/100)*trim_x)
        #pad_y = int((overlap_perc/100)*trim_y)
        #print('pad x, y', pad_x, pad_y)
    
        original_y = trim_y*num_y_segments
        original_x = trim_x*num_x_segments
    
    #not all json logs with start indices have frame_offset info
    try:
        frame_offset = json_chopper_log['frame_offset']
    except:
        frame_offset = 0
    
    #construct image name format for all sub pieces of one image/frame
    #also passes what the stitched image should be named to the stitcher core
    for image in range(num_images):
        
        if is_2D:
            sub_img_format = "_img_{0:03d}".format(image + frame_offset)

            save_name = identifier + sub_img_format + "_annotation.png"       
        
            sub_img_format = "{0}" + sub_img_format + "_x_{1:02d}_y_{2:02d}_annotation.png"
                        
        else:
            sub_img_format = "_frame_{0:03d}".format(image + frame_offset)
    
            save_name = identifier + sub_img_format + "_annotation.png"       
        
            sub_img_format = "{0}_x_{1:02d}_y_{2:02d}" + sub_img_format + "_annotation.png" 

        print("Stitching image {0} of {1}".format(image + frame_offset + 1, num_images))
        overlapping_stitcher_core(pieces_dir, 
                                 save_dir, 
                                 x_start_indices, 
                                 y_start_indices, 
                                 overlapping_x_pix, 
                                 overlapping_y_pix, 
                                 original_x, 
                                 original_y, 
                                 identifier, 
                                 sub_img_format, 
                                 save_name)
    
