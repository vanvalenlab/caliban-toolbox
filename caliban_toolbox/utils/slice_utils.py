# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import numpy as np
import os
import json

from itertools import product

import xarray as xr


def compute_slice_indices(stack_len, slice_len, slice_overlap):
    """ Determine how to slice an image across the stack dimension.

    Args:
        stack_len: total number of z or t stacks
        slice_len: number of z/t frames to be included in each slice
        slice_overlap: number of z/t frames that will overlap in each slice

    Returns:
        numpy.array: coordinates for the start location of each slice
        numpy.array: coordinates for the end location of each slice
    """

    if slice_overlap >= slice_len:
        raise ValueError('slice overlap must be less than the length of the slice')

    spacing = slice_len - slice_overlap

    # slices_start indices begin at index 0, and are spaced 'spacing' apart from one another
    slice_start_indices = np.arange(0, stack_len - slice_overlap, spacing)

    # slices_end indices are 'spacing' away from the start
    slice_end_indices = slice_start_indices + slice_len

    if slice_end_indices[-1] != stack_len:
        # if slices overshoot, reduce length of final slice
        slice_end_indices[-1] = stack_len

    return slice_start_indices, slice_end_indices


def slice_helper(data_xr, slice_start_indices, slice_end_indices):
    """Divide a stack into smaller slices according to supplied indices

    Args:
        data_xr: xarray of to be split into slices
        slice_start_indices: list of indices for where slices start
        slice_end_indices: list of indices for where slices end

    Returns:
        xarray.DataArray: 7D tensor of sliced images
    """

    # get input image dimensions
    fov_len, stack_len, crop_num, input_slice_num, row_len, col_len, chan_len = data_xr.shape

    if input_slice_num > 1:
        raise ValueError('Input array already contains slice data')

    slice_num = len(slice_start_indices)
    sliced_stack_len = slice_end_indices[0] - slice_start_indices[0]

    # create xarray to hold slices
    slice_data = np.zeros((fov_len, sliced_stack_len, crop_num,
                           slice_num, row_len, col_len, chan_len), dtype=data_xr.dtype)

    # labels for each index within a dimension
    coordinate_labels = [data_xr.fovs, range(sliced_stack_len), range(crop_num), range(slice_num),
                         range(row_len), range(col_len), data_xr.channels]

    # labels for each dimension
    dimension_labels = ['fovs', 'stacks', 'crops', 'slices', 'rows', 'cols', 'channels']

    slice_xr = xr.DataArray(data=slice_data, coords=coordinate_labels, dims=dimension_labels)

    # loop through slice indices to generate sliced data
    slice_counter = 0
    for i in range(len(slice_start_indices)):

        if i != len(slice_start_indices) - 1:
            # not the last slice
            slice_xr[:, :, :, slice_counter, ...] = \
                data_xr[:, slice_start_indices[i]:slice_end_indices[i], :, 0, :, :, :].values
            slice_counter += 1

        else:
            # last slice, only index into stack the amount two indices are separated
            slice_len = slice_end_indices[i] - slice_start_indices[i]
            slice_xr[:, :slice_len, :, slice_counter, ...] = \
                data_xr[:, slice_start_indices[i]:slice_end_indices[i], :, 0, :, :, :].values
            slice_counter += 1

    return slice_xr


def stitch_slices(slice_stack, log_data):
    """Helper function to stitch slices together back into original sized array

    Args:
        slice_stack: xarray of shape [fovs, stacks, crops, slices, rows, cols, segmentation_label]
        log_data: log data produced from creation of slice stack

    Returns:
        xarray.DataArray: 7D tensor of stitched labeled slices
    """

    # get parameters from dict
    fov_len, stack_len, crop_num, _, row_len, col_len, chan_len = log_data['original_shape']
    crop_num = log_data.get('num_crops', crop_num)
    row_len = log_data.get('row_crop_size', row_len)
    col_len = log_data.get('col_crop_size', col_len)

    slice_start_indices = log_data['slice_start_indices']
    slice_end_indices = log_data['slice_end_indices']
    num_slices, fov_names = log_data['num_slices'], log_data['fov_names']

    stitched_slices = np.zeros((fov_len, stack_len, crop_num, 1, row_len, col_len, 1),
                               dtype=slice_stack.dtype)

    # loop slice indices to generate sliced data
    for i in range(num_slices - 1):
        stitched_slices[:, slice_start_indices[i]:slice_end_indices[i], :, 0, ...] = \
            slice_stack[:, :, :, i, ...]

    # last slice, only index into stack the amount two indices are separated
    last_idx = num_slices - 1
    slice_len = slice_end_indices[last_idx] - slice_start_indices[last_idx]
    stitched_slices[:, slice_start_indices[last_idx]:slice_end_indices[last_idx], :, 0, ...] = \
        slice_stack[:, :slice_len, :, last_idx, ...]

    return stitched_slices
