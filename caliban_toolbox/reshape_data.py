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

import os
import json

import xarray as xr

from caliban_toolbox.utils import crop_utils, slice_utils, io_utils


def crop_multichannel_data(X_data, y_data, crop_size, overlap_frac, test_parameters=False):
    """Reads in a stack of images and crops them into small pieces for easier annotation

    Args:
        X_data: xarray containing raw images to be cropped
        y_data: xarray containing labeled images to be chopped
        crop_size: (row_crop, col_crop) tuple specifying shape of the crop
        overlap_frac: fraction that crops will overlap each other on each edge
        test_parameters: boolean to determine whether to run all fovs, or only the first

    Returns:
        xarray.DataArray: 7D tensor of cropped data
        dict: relevant data for reconstructing original imaging after cropping
    """

    # sanitize inputs
    if len(crop_size) != 2:
        raise ValueError('crop_size must be a tuple of (row_crop, col_crop), '
                         'got {}'.format(crop_size))

    if not crop_size[0] > 0 and crop_size[1] > 0:
        raise ValueError('crop_size entries must be positive numbers')

    if overlap_frac < 0 or overlap_frac > 1:
        raise ValueError('overlap_frac must be between 0 and 1')

    if list(X_data.dims) != ['fovs', 'stacks', 'crops', 'slices', 'rows', 'cols', 'channels']:
        raise ValueError('X_data does not have expected dims, found {}'.format(X_data.dims))

    if list(y_data.dims) != ['fovs', 'stacks', 'crops', 'slices', 'rows', 'cols', 'channels']:
        raise ValueError('y_data does not have expected dims, found {}'.format(y_data.dims))

    # check if testing or running all samples
    if test_parameters:
        X_data, y_data = X_data[:1, ...], y_data[:1, ...]

    # compute the start and end coordinates for the row and column crops
    row_starts, row_ends, row_padding = crop_utils.compute_crop_indices(img_len=X_data.shape[4],
                                                             crop_size=crop_size[0],
                                                             overlap_frac=overlap_frac)

    col_starts, col_ends, col_padding = crop_utils.compute_crop_indices(img_len=X_data.shape[5],
                                                             crop_size=crop_size[1],
                                                             overlap_frac=overlap_frac)

    # crop images
    X_data_cropped, padded_shape = crop_utils.crop_helper(X_data, row_starts=row_starts, row_ends=row_ends,
                                               col_starts=col_starts, col_ends=col_ends,
                                               padding=(row_padding, col_padding))

    y_data_cropped, padded_shape = crop_utils.crop_helper(y_data, row_starts=row_starts, row_ends=row_ends,
                                               col_starts=col_starts, col_ends=col_ends,
                                               padding=(row_padding, col_padding))

    # save relevant parameters for reconstructing image
    log_data = {}
    log_data['row_starts'] = row_starts.tolist()
    log_data['row_ends'] = row_ends.tolist()
    log_data['row_crop_size'] = crop_size[0]
    log_data['col_starts'] = col_starts.tolist()
    log_data['col_ends'] = col_ends.tolist()
    log_data['col_crop_size'] = crop_size[1]
    log_data['row_padding'] = int(row_padding)
    log_data['col_padding'] = int(col_padding)
    log_data['num_crops'] = X_data_cropped.shape[2]

    return X_data_cropped, y_data_cropped, log_data


def create_slice_data(X_data, y_data, slice_stack_len, slice_overlap=0):
    """Takes an array of data and splits it up into smaller pieces along the stack dimension

    Args:
        X_data: xarray of raw image data to be split
        y_data: xarray of labels to be split
        slice_stack_len: number of z/t frames in each slice
        slice_overlap: number of z/t frames in each slice that overlap one another

    Returns:
        xarray.DataArray: 7D tensor of sliced data
        dict: relevant data for reconstructing original imaging after slicing
    """

    # sanitize inputs
    if len(X_data.shape) != 7:
        raise ValueError('invalid input data shape, '
                         'expected array of len(7), got {}'.format(X_data.shape))

    if len(y_data.shape) != 7:
        raise ValueError('invalid input data shape, '
                         'expected array of len(7), got {}'.format(y_data.shape))

    if slice_stack_len > X_data.shape[1]:
        raise ValueError('slice size is greater than stack length')

    # compute indices for slices
    stack_len = X_data.shape[1]
    slice_start_indices, slice_end_indices = \
        slice_utils.compute_slice_indices(stack_len, slice_stack_len, slice_overlap)

    X_data_slice = slice_utils.slice_helper(X_data, slice_start_indices, slice_end_indices)
    y_data_slice = slice_utils.slice_helper(y_data, slice_start_indices, slice_end_indices)

    log_data = {}
    log_data['slice_start_indices'] = slice_start_indices.tolist()
    log_data['slice_end_indices'] = slice_end_indices.tolist()
    log_data['num_slices'] = len(slice_start_indices)

    return X_data_slice, y_data_slice, log_data


def reconstruct_image_stack(crop_dir, verbose=True):
    """High level function to recombine data into a single stitched image

        Args:
            crop_dir: full path to directory with cropped images
            verbose: flag to control print statements
        """

    # sanitize inputs
    if not os.path.isdir(crop_dir):
        raise ValueError('crop_dir not a valid directory: {}'.format(crop_dir))

    # unpack JSON data
    with open(os.path.join(crop_dir, 'log_data.json')) as json_file:
        log_data = json.load(json_file)

    # combine all npzs into a single stack
    image_stack = io_utils.load_npzs(crop_dir=crop_dir, log_data=log_data, verbose=verbose)

    # stitch slices if data was sliced
    if 'num_slices' in log_data:
        image_stack = slice_utils.stitch_slices(slice_stack=image_stack, log_data=log_data)

    # stitch crops if data was cropped
    if 'num_crops' in log_data:
        image_stack = crop_utils.stitch_crops(crop_stack=image_stack, log_data=log_data)

    # labels for each index within a dimension
    _, stack_len, _, _, row_len, col_len, _ = log_data['original_shape']
    coordinate_labels = [log_data['fov_names'], range(stack_len), range(1),
                         range(1), range(row_len), range(col_len), ['segmentation_label']]

    # labels for each dimension
    dimension_labels = ['fovs', 'stacks', 'crops', 'slices', 'rows', 'cols', 'channels']

    stitched_xr = xr.DataArray(data=image_stack, coords=coordinate_labels,
                               dims=dimension_labels)

    stitched_xr.to_netcdf(os.path.join(crop_dir, 'stitched_images.xr'))
