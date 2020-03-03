# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/caliban-toolbox/LICENSE
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
"""

utility functions for reading/writing files

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import numpy as np
from skimage.io import imread
from skimage.external.tifffile import TiffFile

from caliban_toolbox.utils.misc_utils import sorted_nicely


def get_image(file_name):
    """
    Read image from file and load into numpy array
    """
    ext = os.path.splitext(file_name.lower())[-1]
    if ext == '.tif' or ext == '.tiff':
        return np.float32(TiffFile(file_name).asarray())
    return np.float32(imread(file_name))


def get_img_names(direc_name):
    """
    Return all image filenames in direc_name as sorted list
    """
    imglist = os.listdir(direc_name)
    imgfiles = [i for i in imglist if ".tif" in i or ".png" in i or ".jpg" in i]
    imgfiles = sorted_nicely(imgfiles)
    return imgfiles


def nikon_getfiles(direc_name, channel_name):
    """
    Return all image filenames in direc_name with
    channel_name in the filename
    """
    imglist = os.listdir(direc_name)
    imgfiles = [i for i in imglist if channel_name in i]
    imgfiles = sorted_nicely(imgfiles)
    return imgfiles


def list_npzs_folder(npz_dir):
    '''
    Helper to get all npz names from a given folder. Analogous
    to get_img_names. Filenames are returned in a sorted list.
    Inputs:
        npz_dir: full path to folder that you want to get npz names from
    Output:
        sorted list of npz files
    '''
    all_files = os.listdir(npz_dir)
    npz_list = [i for i in all_files if ".npz" in i]
    npz_list = sorted_nicely(npz_list)
    return npz_list

