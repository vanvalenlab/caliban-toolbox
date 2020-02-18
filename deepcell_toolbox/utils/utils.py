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
Utils functions from deepcell
"""

import os
import re

import numpy as np
from skimage.io import imread
from skimage.external.tifffile import TiffFile
from tensorflow.python.keras import backend as K


def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_image(file_name):
    """
    Read image from file and load into numpy array
    """
    ext = os.path.splitext(file_name.lower())[-1]
    if ext == '.tif' or ext == '.tiff':
        return np.float32(TiffFile(file_name).asarray())
    return np.float32(imread(file_name))


def nikon_getfiles(direc_name, channel_name):
    """
    Return all image filenames in direc_name with
    channel_name in the filename
    """
    imglist = os.listdir(direc_name)
    imgfiles = [i for i in imglist if channel_name in i]
    imgfiles = sorted_nicely(imgfiles)
    return imgfiles


def get_images_from_directory(data_location, channel_names):
    """
    Read all images from directory with channel_name in the filename
    Return them in a numpy array
    """
    data_format = K.image_data_format()
    img_list_channels = []
    for channel in channel_names:
        img_list_channels.append(nikon_getfiles(data_location, channel))

    img_temp = np.asarray(get_image(os.path.join(data_location, img_list_channels[0][0])))

    n_channels = len(channel_names)
    all_images = []

    for stack_iteration in range(len(img_list_channels[0])):
        if data_format == 'channels_first':
            shape = (1, n_channels, img_temp.shape[0], img_temp.shape[1])
        else:
            shape = (1, img_temp.shape[0], img_temp.shape[1], n_channels)

        all_channels = np.zeros(shape, dtype=K.floatx())

        for j in range(n_channels):
            img_path = os.path.join(data_location, img_list_channels[j][stack_iteration])
            channel_img = get_image(img_path)
            if data_format == 'channels_first':
                all_channels[0, j, :, :] = channel_img
            else:
                all_channels[0, :, :, j] = channel_img

        all_images.append(all_channels)

    return all_images
