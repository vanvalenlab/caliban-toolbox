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

from itertools import product

import xarray as xr



def compute_crop_indices(img_len, crop_size=None, crop_num=None, overlap_frac=0):
    """Determine how to crop the image across one dimension.

    Args:
        img_len: length of the image for given dimension
        crop_size: size in pixels of the crop in given dimension; must be specified if
            crop_num not provided
        crop_num: number of crops in the given dimension; must be specified if
            crop_size not provided
        overlap_frac: fraction that adjacent crops will overlap each other on each side

    Returns:
        numpy.array: coordinates for where each crop will start in given dimension
        numpy.array: coordinates for where each crop will end in given dimension
        int: number of pixels of padding at start and end of image in given dimension
    """

    # compute indices based on fixed number of pixels per crop
    if crop_size is not None:

        # compute overlap fraction in pixels
        overlap_pix = math.floor(crop_size * overlap_frac)

    # compute indices based on fixed number of crops
    elif crop_num is not None:
        # number of pixels in non-overlapping portion of crop
        non_overlap_crop_size = np.ceil(img_len / crop_num)

        # Technically this is the fraction the non-overlap, rather than fraction of the whole,
        # but we're going to visually crop overlays anyway to make sure value is appropriate
        overlap_pix = math.floor(non_overlap_crop_size * overlap_frac)

        # total crop size
        crop_size = non_overlap_crop_size + overlap_pix

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


def stitch_crops(crop_stack, log_data):
    """Takes a stack of annotated labels and stitches them together into a single image

    Args:
        crop_stack: 7D tensor of labels to be stitched together
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

    if crop_stack.shape[3] != 1:
        raise ValueError('Stacks must be combined before stitching can occur')

    # for each fov and stack, loop through rows and columns of crop positions
    for fov, stack, row, col in product(range(fov_len), range(stack_len),
                                        range(len(row_starts)), range(len(col_starts))):

        # determine what crop # we're currently working on
        crop_counter = row * len(row_starts) + col

        # get current crop
        crop = crop_stack[fov, stack, crop_counter, 0, :, :, 0]

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

    # trim padding to put image back to original size
    if row_padding > 0:
        stitched_labels = stitched_labels[:, :, :, :, :-row_padding, :, :]
    if col_padding > 0:
        stitched_labels = stitched_labels[:, :, :, :, :, :-col_padding, :]

    return stitched_labels
