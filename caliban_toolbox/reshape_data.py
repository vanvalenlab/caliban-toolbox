# Copyright 2016-2020 David Van Valen at California Institute of Technology
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
    """Determine how to crop the image across one dimension.

    Args:
        img_len: length of the image for given dimension
        crop_size: size in pixels of the crop in given dimension
        overlap_frac: fraction that adjacent crops will overlap each other on each side

    Returns:
        numpy.array: coordinates for where each crop will start in given dimension
        numpy.array: coordinates for where each crop will end in given dimension
        int: number of pixels of padding at start and end of image in given dimension
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

    Args:
        input_data: xarray of [fovs, stacks, crops, slices, rows, cols, channels] to be cropped
        row_starts: list of indices where row crops start
        row_ends: list of indices where row crops end
        col_starts: list of indices where col crops start
        col_ends: list of indices where col crops end
        padding: tuple which specifies the amount of padding on the final image

    Returns:
        numpy.array: 7D tensor of cropped images
        tuple: shape of the final padded image
    """

    # determine key parameters of crop
    fov_len, stack_len, input_crop_num, slice_num, _, _, channel_len = input_data.shape

    if input_crop_num > 1:
        raise ValueError("Array has already been cropped")

    crop_num = len(row_starts) * len(col_starts)
    crop_size_row = row_ends[0] - row_starts[0]
    crop_size_col = col_ends[0] - col_starts[0]

    # create xarray to hold crops
    cropped_stack = np.zeros((fov_len, stack_len, crop_num, slice_num,
                              crop_size_row, crop_size_col, channel_len))

    # labels for each index within a dimension
    coordinate_labels = [input_data.fovs, input_data.stacks, range(crop_num), input_data.slices,
                         range(crop_size_row), range(crop_size_col), input_data.channels]

    # labels for each dimension
    dimension_labels = ['fovs', 'stacks', 'crops', 'slices', 'rows', 'cols', 'channels']

    cropped_xr = xr.DataArray(data=cropped_stack, coords=coordinate_labels, dims=dimension_labels)

    # pad the input to account for imperfectly overlapping final crop in rows and cols
    formatted_padding = ((0, 0), (0, 0), (0, 0), (0, 0), (0, padding[0]), (0, padding[1]), (0, 0))
    padded_input = np.pad(input_data, formatted_padding, mode='constant', constant_values=0)

    # loop through rows and cols to generate crops
    crop_counter = 0
    for i in range(len(row_starts)):
        for j in range(len(col_starts)):
            cropped_xr[:, :, crop_counter, ...] = padded_input[:, :, 0, :,
                                                               row_starts[i]:row_ends[i],
                                                               col_starts[j]:col_ends[j], :]
            crop_counter += 1

    return cropped_xr, padded_input.shape


def crop_multichannel_data(data_xr, crop_size, overlap_frac, test_parameters=False):
    """Reads in a stack of images and crops them into small pieces for easier annotation

    Args:
        data_xr: xarray to be cropped of size [fovs, stacks, 1, slices, rows, cols, channels]
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

    if list(data_xr.dims) != ['fovs', 'stacks', 'crops', 'slices', 'rows', 'cols', 'channels']:
        raise ValueError('data_xr does not have expected dims, found {}'.format(data_xr.dims))

    # check if testing or running all samples
    if test_parameters:
        data_xr = data_xr[:1, ...]

    # compute the start and end coordinates for the row and column crops
    row_starts, row_ends, row_padding = compute_crop_indices(img_len=data_xr.shape[4],
                                                             crop_size=crop_size[0],
                                                             overlap_frac=overlap_frac)

    col_starts, col_ends, col_padding = compute_crop_indices(img_len=data_xr.shape[5],
                                                             crop_size=crop_size[1],
                                                             overlap_frac=overlap_frac)

    # crop images
    data_xr_cropped, padded_shape = crop_helper(data_xr, row_starts=row_starts, row_ends=row_ends,
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
    log_data['num_crops'] = data_xr_cropped.shape[2]

    return data_xr_cropped, log_data


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
                           slice_num, row_len, col_len, chan_len))

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


def create_slice_data(data_xr, slice_stack_len, slice_overlap=0):
    """Takes an array of data and splits it up into smaller pieces along the stack dimension

    Args:
        data_xr: xarray of [fovs, stacks, crops, slices, rows, cols, channels] to be split up
        slice_stack_len: number of z/t frames in each slice
        slice_overlap: number of z/t frames in each slice that overlap one another

    Returns:
        xarray.DataArray: 7D tensor of sliced data
        dict: relevant data for reconstructing original imaging after slicing
    """

    # sanitize inputs
    if len(data_xr.shape) != 7:
        raise ValueError('invalid input data shape, '
                         'expected array of len(7), got {}'.format(data_xr.shape))

    if slice_stack_len > data_xr.shape[1]:
        raise ValueError('slice size is greater than stack length')

    # compute indices for slices
    stack_len = data_xr.shape[1]
    slice_start_indices, slice_end_indices = \
        compute_slice_indices(stack_len, slice_stack_len, slice_overlap)

    slice_xr = slice_helper(data_xr, slice_start_indices, slice_end_indices)

    log_data = {}
    log_data['slice_start_indices'] = slice_start_indices.tolist()
    log_data['slice_end_indices'] = slice_end_indices.tolist()
    log_data['num_slices'] = len(slice_start_indices)

    return slice_xr, log_data


def save_npzs_for_caliban(resized_xr, original_xr, log_data, save_dir, blank_labels='include',
                          save_format='npz', verbose=True):
    """Take an array of processed image data and save as NPZ for caliban

    Args:
        resized_xr: 7D tensor of cropped and sliced data
        original_xr: the unmodified xarray
        log_data: data used to reconstruct images
        save_dir: path to save the npz and JSON files
        blank_labels: whether to include NPZs with blank labels (poor predictions)
                      or skip (no cells)
        save_format: format to save the data (currently only NPZ)
        verbose: flag to control print statements
    """

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # if these are present, it means data was cropped/sliced. Otherwise, default to 1
    num_crops = log_data.get('num_crops', 1)
    num_slices = log_data.get('num_slices', 1)

    fov_names = original_xr.fovs.values
    fov_len = len(fov_names)

    if blank_labels not in ['skip', 'include', 'separate']:
        raise ValueError('blank_labels must be one of '
                         '[skip, include, separate], got {}'.format(blank_labels))

    if blank_labels == 'separate':
        os.makedirs(os.path.join(save_dir, 'separate'))

    # loop through all crops in all images
    for fov in range(fov_len):
        for crop in range(num_crops):
            for slice in range(num_slices):
                # generate identifier for crop
                npz_id = 'fov_{}_crop_{}_slice_{}'.format(fov_names[fov], crop, slice)

                # subset xarray based on supplied indices
                current_xr = resized_xr[fov, :, crop, slice, ...]
                labels = current_xr[..., -1:].values
                channels = current_xr[..., :-1].values

                # determine if labels are blank, and if so what to do with npz
                if np.sum(labels) == 0:

                    # blank labels get saved to separate folder
                    if blank_labels == 'separate':
                        if verbose:
                            print('{} is blank, saving to separate folder'.format(npz_id))
                        save_path = os.path.join(save_dir, blank_labels, npz_id)

                        # save images as either npz or xarray
                        if save_format == 'npz':
                            np.savez(save_path + '.npz', X=channels, y=labels)

                        elif save_format == 'xr':
                            current_xr.to_netcdf(save_path + '.xr')

                    # blank labels don't get saved, empty area of tissue
                    elif blank_labels == 'skip':
                        if verbose:
                            print('{} is blank, skipping saving'.format(npz_id))

                    # blank labels get saved along with other crops
                    elif blank_labels == 'include':
                        if verbose:
                            print('{} is blank, saving to folder'.format(npz_id))
                        save_path = os.path.join(save_dir, npz_id)

                        # save images as either npz or xarray
                        if save_format == 'npz':
                            np.savez(save_path + '.npz', X=channels, y=labels)

                        elif save_format == 'xr':
                            current_xr.to_netcdf(save_path + '.xr')

                else:
                    # crop is not blank, save based on file_format
                    save_path = os.path.join(save_dir, npz_id)

                    # save images as either npz or xarray
                    if save_format == 'npz':
                        np.savez(save_path + '.npz', X=channels, y=labels)

                    elif save_format == 'xr':
                        current_xr.to_netcdf(save_path + '.xr')

    log_data['fov_names'] = fov_names.tolist()
    log_data['channel_names'] = original_xr.channels.values.tolist()
    log_data['original_shape'] = original_xr.shape
    log_data['slice_stack_len'] = resized_xr.shape[1]
    log_data['save_format'] = save_format

    log_path = os.path.join(save_dir, 'log_data.json')
    with open(log_path, 'w') as write_file:
        json.dump(log_data, write_file)


def get_saved_file_path(dir_list, fov_name, crop, slice, file_ext='.npz'):
    """Helper function to identify correct file path for an npz file

    Args:
        dir_list: list of files in directory
        fov_name: string of the current fov_name
        crop: int of current crop
        slice: int of current slice
        file_ext: extension file was saved with

    Returns:
        string: formatted file name

    Raises:
        ValueError: If multiple file path matches were found
    """

    base_string = 'fov_{}_crop_{}_slice_{}'.format(fov_name, crop, slice)
    string_matches = [string for string in dir_list if base_string + '_save_version' in string]

    if len(string_matches) == 0:
        full_string = base_string + file_ext
    elif len(string_matches) == 1:
        full_string = string_matches[0]
    else:
        raise ValueError('Multiple save versions found: '
                         'please select only a single save version. {}'.format(string_matches))
    return full_string


def load_npzs(crop_dir, log_data, verbose=True):
    """Reads all of the cropped images from a directory, and aggregates them into a single stack

    Args:
        crop_dir: path to directory with cropped npz or xarray files
        log_data: dictionary of parameters generated during data saving

        verbose: flag to control print statements

    Returns:
        numpy.array: 7D tensor of labeled crops
    """

    fov_names = log_data['fov_names']
    fov_len, stack_len, _, _, row_size, col_size, _ = log_data['original_shape']
    save_format = log_data['save_format']

    # if cropped/sliced, get size of dimensions. Otherwise, use size in original data
    row_crop_size = log_data.get('row_crop_size', row_size)
    col_crop_size = log_data.get('col_crop_size', col_size)
    slice_stack_len = log_data.get('slice_stack_len', stack_len)

    # if cropped/sliced, get number of crops/slices
    num_crops, num_slices = log_data.get('num_crops', 1), log_data.get('num_slices', 1)
    stack = np.zeros((fov_len, slice_stack_len, num_crops,
                      num_slices, row_crop_size, col_crop_size, 1))
    saved_files = os.listdir(crop_dir)

    # loop through all npz files
    for fov_idx, fov_name in enumerate(fov_names):
        for crop in range(num_crops):
            for slice in range(num_slices):
                # load NPZs
                if save_format == 'npz':
                    npz_path = os.path.join(crop_dir, get_saved_file_path(saved_files, fov_name,
                                                                          crop, slice))
                    if os.path.exists(npz_path):
                        temp_npz = np.load(npz_path)

                        # last slice may be truncated, modify index
                        if slice == num_slices - 1:
                            current_stack_len = temp_npz['X'].shape[1]
                        else:
                            current_stack_len = slice_stack_len

                        stack[fov_idx, :current_stack_len, crop, slice, ...] = temp_npz['y']
                    else:
                        # npz not generated, did not contain any labels, keep blank
                        if verbose:
                            print('could not find npz {}, skipping'.format(npz_path))

                # load xarray
                elif save_format == 'xr':
                    xr_path = os.path.join(crop_dir, get_saved_file_path(saved_files, fov_name,
                                                                         crop, slice))
                    if os.path.exists(xr_path):
                        temp_xr = xr.open_dataarray(xr_path)

                        # last slice may be truncated, modify index
                        if slice == num_slices - 1:
                            current_stack_len = temp_xr.shape[1]
                        else:
                            current_stack_len = stack_len

                        stack[fov_idx, :current_stack_len, crop, slice, ...] = temp_xr[..., -1:]
                    else:
                        # npz not generated, did not contain any labels, keep blank
                        print('could not find xr {}, skipping'.format(xr_path))

    return stack


def stitch_crops(annotated_data, log_data):
    """Takes a stack of annotated labels and stitches them together into a single image

    Args:
        annotated_data: 7D tensor of labels to be stitched together
        log_data: dictionary of parameters for reconstructing original image data

    Returns:
        numpy.array: 7D tensor of reconstructed labels
    """

    # Initialize image with single dimension for channels
    fov_len, stack_len, _, _, row_size, col_size, _ = log_data['original_shape']
    row_padding, col_padding = log_data.get('row_padding', 0), log_data.get('col_padding', 0)
    stitched_labels = np.zeros((fov_len, stack_len, 1, 1, row_size + row_padding,
                                col_size + col_padding, 1))

    row_starts, row_ends = log_data['row_starts'], log_data['row_ends']
    col_starts, col_ends = log_data['col_starts'], log_data['col_ends']

    if annotated_data.shape[3] != 1:
        raise ValueError('Stacks must be combined before stitching can occur')

    # loop through all crops in the stack for each image
    for fov in range(fov_len):
        for stack in range(stack_len):
            crop_counter = 0
            for row in range(len(row_starts)):
                for col in range(len(col_starts)):

                    # get current crop
                    crop = annotated_data[fov, stack, crop_counter, 0, :, :, 0]

                    # increment values to ensure unique labels across final image
                    lowest_allowed_val = np.amax(stitched_labels[fov, stack, ...])
                    crop = np.where(crop == 0, crop, crop + lowest_allowed_val)

                    # get ids of cells in current crop
                    potential_overlap_cells = np.unique(crop)
                    potential_overlap_cells = \
                        potential_overlap_cells[np.nonzero(potential_overlap_cells)]

                    # get values of stitched image at location where crop will be placed
                    stitched_crop = stitched_labels[fov, stack, 0, 0,
                                                    row_starts[row]:row_ends[row],
                                                    col_starts[col]:col_ends[col], 0]

                    # loop through each cell in the crop to determine
                    # if it overlaps with another cell in full image
                    for cell in potential_overlap_cells:

                        # get cell ids present in stitched image
                        # at location of current cell in crop
                        stitched_overlap_vals, stitched_overlap_counts = \
                            np.unique(stitched_crop[crop == cell], return_counts=True)

                        # remove IDs and counts corresponding to overlap with ID 0 (background)
                        keep_vals = np.nonzero(stitched_overlap_vals)
                        stitched_overlap_vals = stitched_overlap_vals[keep_vals]
                        stitched_overlap_counts = stitched_overlap_counts[keep_vals]

                        # if there are overlaps, determine which is greatest in count,
                        # and replace with that ID
                        if len(stitched_overlap_vals) > 0:
                            max_overlap = stitched_overlap_vals[np.argmax(stitched_overlap_counts)]
                            crop[crop == cell] = max_overlap

                    # combine the crop with the current values in the stitched image
                    combined_crop = np.where(stitched_crop > 0, stitched_crop, crop)

                    # use this combined crop to update the values of stitched image
                    stitched_labels[fov, stack, 0, 0, row_starts[row]:row_ends[row],
                                    col_starts[col]:col_ends[col], 0] = combined_crop

                    crop_counter += 1

    # relabel images to remove skipped cell_ids
    return stitched_labels


def reconstruct_image_stack(crop_dir, verbose=True):
    """High level function to combine crops together into a single stitched image

    Args:
        crop_dir: directory where cropped files are stored
        verbose: flag to control print statements
    """

    # sanitize inputs
    if not os.path.isdir(crop_dir):
        raise ValueError('crop_dir not a valid directory: {}'.format(crop_dir))

    # unpack JSON data
    with open(os.path.join(crop_dir, 'log_data.json')) as json_file:
        log_data = json.load(json_file)

    row_padding, col_padding = log_data['row_padding'], log_data['col_padding']
    fov_names = log_data['fov_names']
    # combine all npz crops into a single stack
    crop_stack = load_npzs(crop_dir=crop_dir, log_data=log_data, verbose=verbose)

    # stitch crops together into single contiguous image
    stitched_images = stitch_crops(annotated_data=crop_stack, log_data=log_data)

    # crop image down to original size
    if row_padding > 0:
        stitched_images = stitched_images[:, :, :, :, :-row_padding, :, :]
    if col_padding > 0:
        stitched_images = stitched_images[:, :, :, :, :, :-col_padding, :]

    _, stack_len, _, _, row_len, col_len, _ = log_data['original_shape']

    # labels for each index within a dimension
    coordinate_labels = [fov_names, range(stack_len), range(1),
                         range(1), range(row_len), range(col_len), ['segmentation_label']]

    # labels for each dimension
    dimension_labels = ['fovs', 'stacks', 'crops', 'slices', 'rows', 'cols', 'channels']

    stitched_xr = xr.DataArray(data=stitched_images, coords=coordinate_labels,
                               dims=dimension_labels)

    stitched_xr.to_netcdf(os.path.join(crop_dir, 'stitched_images.nc'))


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

    stitched_slices = np.zeros((fov_len, stack_len, crop_num, 1, row_len, col_len, 1))

    # loop slice indices to generate sliced data
    for i in range(num_slices - 1):
        stitched_slices[:, slice_start_indices[i]:slice_end_indices[i], :, 0, ...] = \
            slice_stack[:, :, :, i, ...]

    # last slice, only index into stack the amount two indices are separated
    last_idx = num_slices - 1
    slice_len = slice_end_indices[last_idx] - slice_start_indices[last_idx]
    stitched_slices[:, slice_start_indices[last_idx]:slice_end_indices[last_idx], :, 0, ...] = \
        slice_stack[:, :slice_len, :, last_idx, ...]

    # labels for each index within a dimension
    coordinate_labels = [fov_names, range(stack_len), range(crop_num), range(1), range(row_len),
                         range(col_len), ['segmentation_label']]

    # labels for each dimension
    dimension_labels = ['fovs', 'stacks', 'crops', 'slices', 'rows', 'cols', 'channels']

    stitched_xr = xr.DataArray(stitched_slices, coords=coordinate_labels, dims=dimension_labels)

    return stitched_xr


def reconstruct_slice_data(save_dir, verbose=True):
    """High level function to put pieces of a slice back together

    Args:
        save_dir: full path to directory where slice pieces are stored
        verbose: flag to control print statements

    Returns:
        xarray.DataArray: 7D tensor of stitched labeled slices
    """

    if not os.path.isdir(save_dir):
        raise FileNotFoundError('slice directory does not exist')

    json_file_path = os.path.join(save_dir, 'log_data.json')
    if not os.path.exists(json_file_path):
        raise FileNotFoundError('json file does not exist')

    with open(json_file_path) as json_file:
        slice_log_data = json.load(json_file)

    slice_stack = load_npzs(save_dir, slice_log_data, verbose=verbose)

    stitched_xr = stitch_slices(slice_stack, slice_log_data)

    return stitched_xr
