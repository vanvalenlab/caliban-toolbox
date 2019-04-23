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
import warnings

from skimage.io import imsave
from deepcell_toolbox.utils.io_utils import get_image, get_img_names
from deepcell_toolbox.utils.misc_utils import sorted_nicely

def overlapping_stitcher(pieces_dir, save_dir, num_x, num_y, overlap_perc, identifier, sub_img_format, save_name):

    #save path
    img_path = os.path.join(save_dir, save_name)
    
    #load test image
    test_img_name = os.listdir(pieces_dir)[0]
    test_img_path = os.path.join(pieces_dir, test_img_name)
    test_img = get_image(test_img_path)
    
    #make empty image, includes zero padding around edges
    
    small_img_size = test_img.shape
    
    trim_y = math.ceil(small_img_size[0]/(1+(2*overlap_perc/100)))
    trim_x = math.ceil(small_img_size[1]/(1+(2*overlap_perc/100)))
    
    pad_x = int((overlap_perc/100)*trim_x)
    pad_y = int((overlap_perc/100)*trim_y)
    
    full_y = trim_y*num_y
    full_x = trim_x*num_x
    
    padded_full_img = np.zeros((full_y + 2*pad_y, full_x + 2*pad_x))

    for j in range(num_y):
        for i in range(num_x):

            #load sub_img
            sub_img_name = sub_img_format.format(identifier,i,j)
            sub_img_path = os.path.join(pieces_dir, sub_img_name)
            sub_img = get_image(sub_img_path)
            
            #don't repeat values even though each sub image has similar values
            lowest_allowed_val = np.amax(padded_full_img)
            sub_img = np.where(sub_img == 0, sub_img, sub_img + lowest_allowed_val)
            
            #if non-zero values in the sub image and the full image overlap, replace all instances of
            #that value in the sub image, keep other values in the sub image the same
            
            #merge top edge of sub_img
            for y_pix in range(2*pad_y):
                for x_pix in range(trim_x+2*pad_x): 

                    sub_img_pix = sub_img[y_pix, x_pix]
                    full_img_pix = padded_full_img[j*trim_y+y_pix, i*trim_x+x_pix]
                    if (sub_img_pix != 0) and (full_img_pix != 0) and (sub_img_pix != full_img_pix):
                        sub_img = np.where(sub_img == sub_img_pix, full_img_pix, sub_img)
            
            #merge left edge of sub_img
            for y_pix in range(trim_y+2*pad_y):
                for x_pix in range(2*pad_x):
                    sub_img_pix = sub_img[y_pix, x_pix]
                    full_img_pix = padded_full_img[j*trim_y+y_pix, i*trim_x+x_pix]
                    if (sub_img_pix != 0) and (full_img_pix != 0) and (sub_img_pix != full_img_pix):
                        sub_img = np.where(sub_img == sub_img_pix, full_img_pix, sub_img)
            
            #put sub image into the full image
            padded_full_img[int(j*trim_y):int(((j+1) * trim_y) + (2*pad_y)),
                                             int(i*trim_x):int(((i+1) * trim_x) + (2*pad_x))] = sub_img
            
    padded_full_img = padded_full_img.astype(np.uint16)

    #trim off edges
    full_img = padded_full_img[pad_y:pad_y+full_y, pad_x:pad_x+full_x]

    #save big image
    #suppress "low contrast image" warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(img_path, full_img)
