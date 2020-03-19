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
import json


import xarray as xr


def compute_crop_indices(img_len, crop_size, overlap_frac):
    """ Determine how to crop the image across one dimension.

    Inputs
        img_len: length of the image for given dimension
        crop_size: size in pixels of the crop in given dimension
        overlap_frac: fraction that adjacent crops will overlap each other on each side

    Outputs:
        start_indices: array of coordinates for where each crop will start in given dimension
        end_indices: array of coordinates for where each crop will end in given dimension
        padding: number of pixels of padding at start and end of image in given dimension
    """

    # compute overlap fraction in pixels
    overlap_pix = math.floor(crop_size * overlap_frac)

    # the crops start at pixel 0, and are spaced crop_size - overlap_pix away from each other
    start_indices = np.arange(0, img_len - overlap_pix, crop_size - overlap_pix)

    # the crops each end crop_size away the start
    end_indices = start_indices + crop_size

    # the padding for the final image is the amount that the last crop goes beyond the image size
    padding = end_indices[-1] - img_len

    return start_indices, end_indices, padding


def crop_helper(input_data, row_starts, row_ends, col_starts, col_ends, padding):
    """Crops an image into pieces according to supplied coordinates

    Inputs
        input_data: xarray of [fovs, stacks, crops, slices, rows, cols, channels] to be cropped
        row_starts: list of indices where row crops start
        row_ends: list of indices where row crops end
        col_starts: list of indices where col crops start
        col_ends: list of indices where col crops end
        padding: tuple of (row_pad, col_pad) which specifies the amount of padding to add the final image

    Outputs:
        cropped_stack: stack of cropped images of
            shape [fovs, stacks, crops, slices, cropped_rows, cropped_cols, channels]
        padded_image_shape: shape of the final padded image"""

    # determine key parameters of crop
    fov_len, stack_len, input_crop_num, slice_num, _, _, channel_len = input_data.shape

    if input_crop_num > 1:
        raise ValueError("Array has already been cropped")

    crop_num = len(row_starts) * len(col_starts)
    crop_size_row = row_ends[0] - row_starts[0]
    crop_size_col = col_ends[0] - col_starts[0]

    # create xarray to hold crops
    cropped_stack = np.zeros((fov_len, stack_len, crop_num, slice_num, crop_size_row, crop_size_col, channel_len))
    cropped_xr = xr.DataArray(data=cropped_stack, coords=[input_data.fovs, input_data.stacks, range(crop_num),
                                                          input_data.slices, range(crop_size_row),
                                                          range(crop_size_col), input_data.channels],
                              dims=["fovs", "stacks", "crops", "slices", "rows", "cols", "channels"])

    # pad the input to account for imperfectly overlapping final crop in rows and cols
    formatted_padding = ((0, 0), (0, 0), (0, 0), (0, 0), (0, padding[0]), (0, padding[1]), (0, 0))
    padded_input = np.pad(input_data, formatted_padding, mode="constant", constant_values=0)

    # loop through rows and cols to generate crops
    crop_counter = 0
    for i in range(len(row_starts)):
        for j in range(len(col_starts)):
            cropped_xr[:, :, crop_counter, ...] = padded_input[:, :, 0, :, row_starts[i]:row_ends[i], col_starts[j]:col_ends[j], :]
            crop_counter += 1

    return cropped_xr, padded_input.shape


def crop_multichannel_data(data_xr, crop_size, overlap_frac, test_parameters=False):
    """Reads in a stack of images and crops them into small pieces for easier annotation

    Inputs
        data_xr: xarray to be cropped of size [fovs, stacks, 1, slices, rows, cols, channels]
        crop_size: (row_crop, col_crop) tuple specifying shape of the crop
        overlap_frac: fraction that crops will overlap each other on each edge
        test_parameters: boolean to determine whether to run all fovs and save to disk, or only first and return values

    Outputs:
        data_xr_cropped: xarray of [fovs, stacks, crops, slices, rows_cropped, cols_cropped, channels """

    # sanitize inputs
    if len(crop_size) != 2:
        raise ValueError("crop_size must be a tuple of (row_crop, col_crop), got {}".format(crop_size))

    if not crop_size[0] > 0 and crop_size[1] > 0:
        raise ValueError("crop_size entries must be positive numbers")

    if overlap_frac < 0 or overlap_frac > 1:
        raise ValueError("overlap_frac must be between 0 and 1")

    if list(data_xr.dims) != ["fovs", "stacks", "crops", "slices", "rows", "cols", "channels"]:
        raise ValueError("data_xr does not have expected dims, found {}".format(data_xr.dims))

    # check if testing or running all samples
    if test_parameters:
        data_xr = data_xr[:1, ...]

    # compute the start and end coordinates for the row and column crops
    row_starts, row_ends, row_padding = compute_crop_indices(img_len=data_xr.shape[4], crop_size=crop_size[0],
                                                             overlap_frac=overlap_frac)

    col_starts, col_ends, col_padding = compute_crop_indices(img_len=data_xr.shape[5], crop_size=crop_size[1],
                                                             overlap_frac=overlap_frac)

    # crop images
    data_xr_cropped, padded_shape = crop_helper(data_xr, row_starts=row_starts, row_ends=row_ends,
                                                col_starts=col_starts, col_ends=col_ends,
                                                padding=(row_padding, col_padding))

    # save relevant parameters for reconstructing image
    log_data = {}
    log_data["row_starts"] = row_starts.tolist()
    log_data["row_ends"] = row_ends.tolist()
    log_data["row_crop_size"] = crop_size[0]
    log_data["num_row_crops"] = len(row_starts)
    log_data["col_starts"] = col_starts.tolist()
    log_data["col_ends"] = col_ends.tolist()
    log_data["col_crop_size"] = crop_size[1]
    log_data["num_col_crops"] = len(col_starts)
    log_data["row_padding"] = int(row_padding)
    log_data["col_padding"] = int(col_padding)
    log_data["num_crops"] = data_xr_cropped.shape[2]

    return data_xr_cropped, log_data


def compute_slice_indices(stack_len, slice_len, slice_overlap):
    """ Determine how to slice an image across the stack dimension.

    Inputs
        stack_len: total number of z or t stacks
        slice_len: number of z/t frames to be included in each slice
        slice_overlap: number of z/t frames that will overlap in each slice

    Outputs:
        slice_start_indices: array of coordinates for the start location of each slice
        slice_end_indices: array of coordinates for the start location of each slice """

    if slice_overlap >= slice_len:
        raise ValueError("slice overlap must be less than the length of the slice")

    spacing = slice_len - slice_overlap

    # slices_start indices begin at index 0, and are spaced "spacing" apart from one another
    slice_start_indices = np.arange(0, stack_len - slice_overlap, spacing)

    # slices_end indices are "spacing" away from the start
    slice_end_indices = slice_start_indices + slice_len

    if slice_end_indices[-1] != stack_len:
        # if slices overshoot, reduce length of final slice
        slice_end_indices[-1] = stack_len

    return slice_start_indices, slice_end_indices


def slice_helper(data_xr, slice_start_indices, slice_end_indices):
    """Divide a stack into smaller slices according to supplied indices

    Inputs
        data_stack: xarray of [fovs, stacks, crops, slices, rows, cols, channels] to be split into slices
        slice_start_indices: list of indices for where slices start
        slice_end_indices: list of indices for where slices end

    Outputs:
        slice_xr: xarray of sliced images of [fovs, stacks, crops, slices, rows, cols, channels]"""

    # get input image dimensions
    fov_len, stack_len, crop_num, input_slice_num, row_len, col_len, chan_len = data_xr.shape

    if input_slice_num > 1:
        raise ValueError("Input array already contains slice data")

    slice_num = len(slice_start_indices)
    sliced_stack_len = slice_end_indices[0] - slice_start_indices[0]

    # create xarray to hold slices
    slice_data = np.zeros((fov_len, sliced_stack_len, crop_num, slice_num, row_len, col_len, chan_len))
    slice_xr = xr.DataArray(data=slice_data, coords=[data_xr.fovs, range(sliced_stack_len), range(crop_num),
                                                     range(slice_num), range(row_len), range(col_len),
                                                     data_xr.channels],
                              dims=["fovs", "stacks", "crops", "slices", "rows", "cols", "channels"])

    # loop through slice indices to generate sliced data
    slice_counter = 0
    for i in range(len(slice_start_indices)):

        if i != len(slice_start_indices) - 1:
            # not the last slice
            slice_xr[:, :, :, slice_counter, ...] = data_xr[:, slice_start_indices[i]:slice_end_indices[i],
                                                                :, 0, :, :, :].values
            slice_counter += 1

        else:
            # last slice, only index into stack the amount two indices are separated
            slice_len = slice_end_indices[i] - slice_start_indices[i]
            slice_xr[:, :slice_len, :, slice_counter, ...] = data_xr[:, slice_start_indices[i]:slice_end_indices[i], :, 0, :, :, :].values
            slice_counter += 1

    return slice_xr


def create_slice_data(data_xr, slice_stack_len, slice_overlap=0):
    """Takes an array of data and splits it up into smaller pieces along the stack dimension

    Inputs
        data_xr: xarray of [fovs, stacks, crops, slices, rows, cols, channels] to be split up
        slice_stack_len: number of z/t frames in each slice
        slice_overlap: number of z/t frames in each slice that overlap one another

    Outputs
        slice_xr: xarray of [fovs, stacks, crops, slices, rows, cols, channels] that has been split
        log_data: dictionary containing data for reconstructing original image"""

    # sanitize inputs
    if len(data_xr.shape) != 7:
        raise ValueError("invalid input data shape, expected array of len(7), got {}".format(data_xr.shape))

    if slice_stack_len > data_xr.shape[1]:
        raise ValueError("slice size is greater than stack length")

    # compute indices for slices
    stack_len = data_xr.shape[1]
    slice_start_indices, slice_end_indices = compute_slice_indices(stack_len, slice_stack_len, slice_overlap)

    slice_xr = slice_helper(data_xr, slice_start_indices, slice_end_indices)

    log_data = {}
    log_data["slice_start_indices"] = slice_start_indices.tolist()
    log_data["slice_end_indices"] = slice_end_indices.tolist()
    log_data["num_slices"] = len(slice_start_indices)

    return slice_xr, log_data


def save_npzs_for_caliban(resized_xr, original_xr, log_data,  save_dir, blank_labels="include", save_format="npz"):
    """Take an array of processed image data and save as NPZ for caliban

    Inputs
        resized_xr: xarray of [fovs, stacks, crops, slices, rows, cols, channels] that has been reshaped
        original_xr: the unmodified xarray
        log_data: data used to reconstruct images
        save_dir: path to save the npz and JSON files

    Outputs
        None (saves npz and JSON to disk)"""

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    num_row_crops = log_data.get("num_row_crops", 1)
    num_col_crops = log_data.get("num_col_crops", 1)
    num_slices = log_data.get("num_slices", 1)

    fov_names = original_xr.fovs.values
    fov_len = len(fov_names)

    if blank_labels not in ["skip", "include", "separate"]:
        raise ValueError("blank_labels must be one of ['skip', 'include', 'separate'], got {}".format(blank_labels))

    if blank_labels == "separate":
        os.makedirs(os.path.join(save_dir, "separate"))

    # loop through all crops in all images
    for fov in range(fov_len):
        crop_counter = 0
        for row in range(num_row_crops):
            for col in range(num_col_crops):
                for slice in range(num_slices):
                    # generate identifier for crop
                    npz_id = "fov_{}_row_{}_col_{}_slice_{}".format(fov_names[fov], row, col, slice)

                    # subset xarray based on supplied indices
                    current_xr = resized_xr[fov:(fov + 1), :, crop_counter, slice,  ...]
                    labels = current_xr[..., -1:].values
                    channels = current_xr[..., :-1].values

                    # determine if labels are blank, and if so what to do with npz
                    if np.sum(labels) == 0:

                        # blank labels get saved to separate folder
                        if blank_labels == "separate":
                            print("{} is blank, saving to separate folder".format(npz_id))
                            save_path = os.path.join(save_dir, blank_labels, npz_id)

                            # save images as either npz or xarray
                            if save_format == 'npz':
                                np.savez(save_path + ".npz", X=channels, y=labels)

                            elif save_format == 'xr':
                                current_xr.to_netcdf(save_path + ".xr")

                        # blank labels don't get saved, empty area of tissue
                        elif blank_labels == "skip":
                            print("{} is blank, skipping saving".format(npz_id))

                        # blank labels get saved along with other crops
                        elif blank_labels == "include":
                            print("{} is blank, saving to folder".format(npz_id))
                            save_path = os.path.join(save_dir, npz_id)

                            # save images as either npz or xarray
                            if save_format == 'npz':
                                np.savez(save_path + ".npz", X=channels, y=labels)

                            elif save_format == 'xr':
                                current_xr.to_netcdf(save_path + ".xr")

                    else:
                        # crop is not blank, save based on file_format
                        save_path = os.path.join(save_dir, npz_id)

                        # save images as either npz or xarray
                        if save_format == 'npz':
                            np.savez(save_path + ".npz", X=channels, y=labels)

                        elif save_format == 'xr':
                            current_xr.to_netcdf(save_path + ".xr")

                crop_counter += 1

    log_data["fov_names"] = fov_names.tolist()
    log_data["channel_names"] = original_xr.channels.values.tolist()
    log_data["original_shape"] = original_xr.shape
    log_data["slice_stack_len"] = resized_xr.shape[1]
    log_data["save_format"] = save_format

    log_path = os.path.join(save_dir, "log_data.json")
    with open(log_path, "w") as write_file:
        json.dump(log_data, write_file)
