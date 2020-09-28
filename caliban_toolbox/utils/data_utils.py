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
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import xarray as xr


def pad_xr_dims(input_xr, padded_dims=None):
    """Takes an xarray and pads it with dimensions of size 1 according to the supplied dims list

    Inputs
        input_xr: xarray to pad
        padded_dims: ordered list of final dims; new dims will be added with size 1. If None,
            defaults to standard naming scheme for pipeline

    Outputs
        padded_xr: xarray that has had additional dims added of size 1

    Raises:
        ValueError: If padded dims includes existing dims in a different order
        ValueError: If padded dims includes duplicate names
    """

    if padded_dims is None:
        padded_dims = ["fovs", "stacks", "crops", "slices", "rows", "cols", "channels"]
    # make sure that dimensions which are present in both lists are in same order
    old_dims = [dim for dim in padded_dims if dim in input_xr.dims]

    if not old_dims == list(input_xr.dims):
        raise ValueError("existing dimensions in the xarray must be in same order")

    if len(np.unique(padded_dims)) != len(padded_dims):
        raise ValueError('Dimensions must have unique names')

    # create new output data
    output_vals = input_xr.values
    output_coords = []

    for idx, dim in enumerate(padded_dims):

        if dim in input_xr.dims:
            # dimension already exists, using existing values and coords
            output_coords.append(input_xr[dim])
        else:
            output_vals = np.expand_dims(output_vals, axis=idx)
            output_coords.append(range(1))

    padded_xr = xr.DataArray(output_vals, coords=output_coords, dims=padded_dims)

    return padded_xr


def create_blank_channel(img_size, dtype='int16', full_blank=False):
    """Creates a mostly blank channel of specified size

    Args:
        img_size: tuple specifying the size of the image to create
        dtype: dtype for image
        full_blank: boolean to set whether image has few sparse pixels, or is completely blank

    Returns:
        numpy.array: a (mostly) blank array with positive pixels in random values
    """

    blank = np.zeros(img_size, dtype=dtype)

    if full_blank:
        return blank
    else:
        # noise will be created within 100 pixel boxes
        row_steps = math.floor(blank.shape[0] / 100)
        col_steps = math.floor(blank.shape[1] / 100)

        for row_step in range(row_steps):
            for col_step in range(col_steps):
                row_index = np.random.randint(0, 100 - 1)
                col_index = np.random.randint(0, 100 - 1)
                blank[row_step * 100 + row_index, col_step * 100 + col_index] = \
                    np.random.randint(1, 15)

        return blank


def reorder_channels(new_channel_order, input_data, full_blank=True):
    """Reorders the channels in an xarray to match new_channel_order. New channels will be blank

    Args:
        new_channel_order: ordered list of channels for output data
        input_data: xarray to be reordered
        full_blank: whether new channels should be completely blank (for visualization),
            or mostly blank with noise (for model training to avoid divide by zero errors).

    Returns:
        xarray.DataArray: Reordered version of input_data

    Raises:
        ValueError: If new_channel_order contains duplicated entries
    """

    # error checking
    vals, counts = np.unique(new_channel_order, return_counts=True)
    duplicated = np.where(counts > 1)
    if len(duplicated[0] > 0):
        raise ValueError("The following channels are duplicated "
                         "in  new_channel_order: {}".format(vals[duplicated[0]]))

    # create array for output data
    full_array = np.zeros((input_data.shape[:-1] + (len(new_channel_order),)),
                          dtype=input_data.dtype)

    existing_channels = input_data.channels

    for i in range(len(new_channel_order)):
        if new_channel_order[i] in existing_channels:
            current_channel = input_data.loc[:, :, :, new_channel_order[i]].values
            full_array[:, :, :, i] = current_channel
        else:
            print('Creating blank channel with {}'.format(new_channel_order[i]))
            blank = create_blank_channel(input_data.shape[1:3], dtype=input_data.dtype,
                                         full_blank=full_blank)
            full_array[:, :, :, i] = blank

    coords = [input_data.fovs, range(input_data.shape[1]),
              range(input_data.shape[2]), new_channel_order]

    dims = ["fovs", "rows", "cols", "channels"]

    channel_xr_blanked = xr.DataArray(full_array, coords=coords, dims=dims)

    return channel_xr_blanked


def make_blank_labels(image_data, dtype='uint16'):
    """Creates an xarray of blank y_labels which matches the image_data passed in

    Args:
        image_data: xarray of image channels used to get label names
        dtype: dtype for labels

    Returns:
        xarray.DataArray: blank xarray of labeled data
    """

    blank_data = np.zeros(image_data.shape[:-1] + (1,), dtype=dtype)

    coords = [image_data.fovs, image_data.rows, image_data.cols, ['segmentation_label']]
    blank_xr = xr.DataArray(blank_data, coords=coords, dims=image_data.dims)

    return blank_xr
