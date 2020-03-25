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

import numpy as np
import os
import json

import xarray as xr

from segmentation.utils import data_utils


def get_saved_file_path(dir_list, fov_name, row, col, slice, file_ext=".npz"):
    """Helper function to identify correct file path for an npz file

    Inputs
        dir_list: list of files in directory
        fov_name: string of the current fov_name
        row: int of current row
        col: int of current col
        slice: int of current slice
        file_ext: extension file was saved with

    Returns
        file_path: string of formatted file path with appropriate ending"""

    base_string = "fov_{}_row_{}_col_{}_slice_{}".format(fov_name, row, col, slice)
    string_matches = [string for string in dir_list if base_string + "_save_version" in string]

    if len(string_matches) == 0:
        full_string = base_string + file_ext
    elif len(string_matches) == 1:
        full_string = string_matches[0]
    else:
        raise ValueError("Multiple save versions found: "
                         "please select only a single save version. {}".format(string_matches))
    return full_string


def load_npzs(crop_dir, log_data, verbose=True):
    """Reads all of the cropped images from a directory, and aggregates them into a single stack

    Inputs:
        crop_dir: path to directory with cropped npz or xarray files
        log_data: dictionary of parameters generated during data saving
        verbose: flag to control print statements

    Outputs:
        stack: combined array of all labeled images"""

    fov_names = log_data["fov_names"]
    num_crops, num_slices = log_data.get("num_crops", 1), log_data.get("num_slices", 1)
    num_row_crops, num_col_crops = log_data.get("num_row_crops", 1), log_data.get("num_col_crops", 1)
    fov_len, stack_len, _, _, row_size, col_size, _ = log_data["original_shape"]
    slice_stack_len = log_data.get("slice_stack_len", stack_len)

    row_crop_size, col_crop_size = log_data.get("row_crop_size", row_size), log_data.get("col_crop_size", col_size)
    save_format = log_data["save_format"]

    stack = np.zeros((fov_len, slice_stack_len, num_crops, num_slices, row_crop_size, col_crop_size, 1))
    saved_files = os.listdir(crop_dir)

    # loop through all npz files
    for fov_idx, fov_name in enumerate(fov_names):
        crop_idx = 0
        for row in range(num_row_crops):
            for col in range(num_col_crops):
                for slice in range(num_slices):

                    # load NPZs
                    if save_format == "npz":
                        npz_path = os.path.join(crop_dir, get_saved_file_path(saved_files, fov_name, row, col, slice))
                        if os.path.exists(npz_path):
                            temp_npz = np.load(npz_path)

                            # last slice may be truncated, modify index
                            if slice == num_slices - 1:
                                current_stack_len = temp_npz["X"].shape[1]
                            else:
                                current_stack_len = slice_stack_len

                            stack[fov_idx:(fov_idx + 1), :current_stack_len, crop_idx, slice, ...] = temp_npz["y"]
                        else:
                            # npz not generated, did not contain any labels, keep blank
                            if verbose:
                                print("could not find npz {}, skipping".format(npz_path))

                    # load xarray
                    elif save_format == "xr":
                        xr_path = os.path.join(crop_dir, get_saved_file_path(saved_files, fov_name, row, col, slice))
                        if os.path.exists(xr_path):
                            temp_xr = xr.open_dataarray(xr_path)

                            # last slice may be truncated, modify index
                            if slice == num_slices - 1:
                                current_stack_len = temp_xr.shape[1]
                            else:
                                current_stack_len = stack_len

                            stack[fov_idx:(fov_idx + 1), :current_stack_len, crop_idx, slice, ...] = temp_xr[..., -1:]
                        else:
                            # npz not generated, did not contain any labels, keep blank
                            print("could not find xr {}, skipping".format(xr_path))

                crop_idx += 1

    return stack


def stitch_crops(annotated_data, log_data):
    """Takes a stack of annotated labels and stitches them together into a single image

    Inputs:
        annotated_data: xr of [fovs, stacks, crop_num, slice_num, rows, cols, channels] to be stitched together
        log_data: dictionary of parameters for reconstructing original image data

    Outputs:
        stitched_image: stitched labels of [fovs, stacks, crops, slices, rows, cols, channels] """

    # Initialize image with single dimension for channels
    fov_len, stack_len, _, _, row_size, col_size, _ = log_data["original_shape"]
    row_padding, col_padding = log_data.get("row_padding", 0), log_data.get("col_padding", 0)
    stitched_labels = np.zeros((fov_len, stack_len, 1, 1, row_size + row_padding, col_size + col_padding, 1))

    row_starts, row_ends = log_data["row_starts"], log_data["row_ends"]
    col_starts, col_ends = log_data["col_starts"], log_data["col_ends"]

    if annotated_data.shape[3] != 1:
        raise ValueError("Stacks must be combined before stitching can occur")

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
                    potential_overlap_cells = potential_overlap_cells[np.nonzero(potential_overlap_cells)]

                    # get values of stitched image at location where crop will be placed
                    stitched_crop = stitched_labels[fov, stack, 0, 0, row_starts[row]:row_ends[row], col_starts[col]:col_ends[col], 0]

                    # loop through each cell in the crop to determine if it overlaps with another cell in full image
                    for cell in potential_overlap_cells:

                        # get cell ids present in stitched image at location of current cell in crop
                        stitched_overlap_vals, stitched_overlap_counts = np.unique(stitched_crop[crop == cell],
                                                                                   return_counts=True)

                        # remove IDs and counts corresponding to overlap with ID 0 (background)
                        keep_vals = np.nonzero(stitched_overlap_vals)
                        stitched_overlap_vals = stitched_overlap_vals[keep_vals]
                        stitched_overlap_counts = stitched_overlap_counts[keep_vals]

                        # if there are overlaps, determine which is greatest in count, and replace with that ID
                        if len(stitched_overlap_vals) > 0:
                            max_overlap = stitched_overlap_vals[np.argmax(stitched_overlap_counts)]
                            crop[crop == cell] = max_overlap

                    # combine the crop with the current values in the stitched image
                    combined_crop = np.where(stitched_crop > 0, stitched_crop, crop)

                    # use this combined crop to update the values of stitched image
                    stitched_labels[fov, stack, 0, 0, row_starts[row]:row_ends[row], col_starts[col]:col_ends[col], 0] = combined_crop

                    crop_counter += 1

    # relabel images to remove skipped cell_ids
    return stitched_labels


def reconstruct_image_stack(crop_dir, verbose=True):
    """High level function to combine crops together into a single stitched image

    Inputs:
        crop_dir: directory where cropped files are stored
        verbose: flag to control print statements

    Outputs:
        None (saves stitched xarray to folder)"""

    # sanitize inputs
    if not os.path.isdir(crop_dir):
        raise ValueError("crop_dir not a valid directory: {}".format(crop_dir))

    # unpack JSON data
    with open(os.path.join(crop_dir, "log_data.json")) as json_file:
        log_data = json.load(json_file)

    row_padding, col_padding = log_data["row_padding"], log_data["col_padding"]
    fov_names, channel_names = log_data["fov_names"], log_data["channel_names"]
    # combine all npz crops into a single stack
    crop_stack = load_npzs(crop_dir=crop_dir, log_data=log_data, verbose=verbose)

    # stitch crops together into single contiguous image
    stitched_images = stitch_crops(annotated_data=crop_stack, log_data=log_data)

    # crop image down to original size
    if row_padding > 0:
        stitched_images = stitched_images[:, :, :, :, :-row_padding, :, :]
    if col_padding > 0:
        stitched_images = stitched_images[:, :, :, :, :, :-col_padding, :]

    _, stack_len, _, _, row_len, col_len, _ = log_data["original_shape"]
    stitched_xr = xr.DataArray(data=stitched_images,
                               coords=[fov_names, range(stack_len), range(1), range(1), range(row_len), range(col_len),
                                       ["segmentation_label"]], dims=["fovs", "stacks", "crops", "slices",
                                                                      "rows", "cols", "channels"])

    stitched_xr.to_netcdf(os.path.join(crop_dir, "stitched_images.nc"))


def overlay_grid_lines(overlay_img, row_starts, row_ends, col_starts, col_ends):
    """Visualize the location of image crops on the original uncropped image to assess crop size

    Inputs
        overlay_img: original image [rows x cols] that crops will overlaid onto
        row_start:s vector of start indices generated by crop_multichannel_data
        row_ends: vector of end indices generated by crop_multichannel_data
        col_starts: vector of start indices generated by crop_multichannel_data
        col_ends: vector of end indices generated by crop_multichannel_data

    Outputs
        overlay_img: original image overlaid with the crop borders"""

    # get dimensions of the image
    row_len = overlay_img.shape[0]
    col_len = overlay_img.shape[1]

    # if first start position is 0, we won't plot it since it's on the image border
    if row_starts[0] == 0:
        row_starts = row_starts[1:]
    if col_starts[0] == 0:
        col_starts = col_starts[1:]

    # use distance between start index of crop 1 and end index of crop 0 determine overlap amount
    row_sep = row_ends[0] - row_starts[0]
    col_sep = col_ends[0] - col_starts[0]

    # generate a vector of alternating 0s and im_max for a dotted line
    val = np.max(overlay_img)
    dotted = [0, 0, 0, 0, val, val, val, val]
    side = np.max(overlay_img.shape)
    dotted = dotted * side

    # trim dotted vectors to be same length as respective image dimension
    row_dotted = np.expand_dims(np.array(dotted[:row_len]), axis=-1)
    col_dotted = np.expand_dims(np.array(dotted[:col_len]), axis=0)

    # expand the thickness to 3 pixel width for better visualization
    row_starts = row_starts + [x + 1 for x in row_starts] + [x + 2 for x in row_starts]
    row_ends = [x + row_sep for x in row_starts]

    # define the location of each image to be halfway between the overlap boundary on either side
    row_middle = [int(x + (row_sep / 2)) for x in row_starts]

    # same for columns
    col_starts = col_starts + [x + 1 for x in col_starts] + [x + 2 for x in col_starts]
    col_middle = [int(x + (col_sep / 2)) for x in col_starts]
    col_ends = [x + col_sep for x in col_starts]

    # set the values of start and end indices to be dotted lines
    overlay_img[row_starts, :] = col_dotted
    overlay_img[row_ends, :] = col_dotted
    overlay_img[:, col_starts] = row_dotted
    overlay_img[:, col_ends] = row_dotted

    # set the values of the line delineating image crop to be constant value
    overlay_img[row_middle, :] = val
    overlay_img[:, col_middle] = val

    return overlay_img


def overlay_crop_overlap(img_crop, row_starts, row_ends, col_starts, col_ends):
    """Visualize the extent of overlap between adjacent crops by plotting overlap regions on an example crop

    Inputs
        img_crop: example crop generated by crop_multichannel_data over which overlap will be plotted
        row_starts: vector of start indices generated by crop_multichannel_data
        row_ends: vector of end indices generated by crop_multichannel_data
        col_starts: vector of start indices generated by crop_multichannel_data
        col_ends: vector of end indices generated by crop_multichannel_data

    Outputs
        img_crop: example crop with dotted lines superimposed on location of adjacent overlapping crops"""

    # get image dimensions
    row_len, col_len = img_crop.shape[0], img_crop.shape[1]

    # compute amount of overlap on each axis
    row_overlap = row_ends[0] - row_starts[1]
    col_overlap = col_ends[0] - col_starts[1]

    # generate vector to super-impose dotted line
    val = np.max(img_crop)
    dotted = [0, 0, 0, 0, val, val, val, val]
    side = np.max(img_crop.shape)
    dotted = dotted * side

    # trim dotted vectors to be same length as respective sides of image
    row_dotted = dotted[:row_len]
    col_dotted = np.expand_dims(np.array(dotted[:col_len]), axis=-1)

    # overlay the dotted vectors on the original image at locations of overlap
    img_crop[[row_overlap, row_len - row_overlap], :] = row_dotted
    img_crop[:, [col_overlap, col_len - col_overlap]] = col_dotted

    return img_crop


def set_channel_colors(combined_xr, plot_colors):
    """Modifies the order of channels in xarray so they're displayed with appropriate color in caliban

    Inputs
        combined_xr: xarray containing channels and labels
        plot_colors: array containing the color of each channel, in order of the current channels

    Returns
        reordered_xr: xarray containing the channels and labels reordered and filled with blanks
                      to enable visualization in caliban"""

    # first define the order that channels are visualize
    channel_order = np.array(["red", "green", "blue", "cyan", "magenta", "yellow", "segmentation_label"])

    # create the array holding the final ordering of channel names
    final_channel_names = np.array(["red", "green", "blue", "cyan", "magenta", "yellow", "segmentation_label"],
                                   dtype="<U20")

    # make sure supplied plot_colors exist as available channels
    if not np.all(np.isin(plot_colors, channel_order)):
        raise ValueError("supplied plot_colors not valid, must be one of: {}".format(channel_order[:-1]))

    # make sure all imaging channels have a plot_color
    if len(plot_colors) != combined_xr.shape[-1] - 1:
        raise ValueError("Mismatch between number of imaging channels and supplied plot colors")

    channel_names = combined_xr.channels.values

    # loop through each of the supplied plot colors
    for idx in range(len(plot_colors)):
        # get the position of that plot color in the channel order
        final_idx = np.isin(channel_order, plot_colors[idx])

        # assign the channel corresponding to that color to that position in the final ordering
        final_channel_names[final_idx] = channel_names[idx]

    # figure out which channels contain real data, don't substitute these with a blank tif
    non_blank_channels = final_channel_names[np.isin(final_channel_names, combined_xr.channels.values)]

    # reorder the xarray
    reordered_xr = data_utils.reorder_xarray_channels(channel_order=final_channel_names, channel_xr=combined_xr,
                                                      non_blank_channels=non_blank_channels)

    return reordered_xr


def stitch_slices(slice_stack, log_data):
    """Helper function to stitch slices together back into original sized array

    Inputs
        slice_stack: xarray of shape [fovs, stacks, crops, slices, rows, cols, segmentation_label]
        log_data: log data produced from creation of slice stack

    Outputs
        stitched_slices: xarray of shape [fovs, stacks, crops, slices, rows, cols, segmentation_label]"""

    # get parameters from dict
    fov_len, stack_len, crop_num, _, row_len, col_len, chan_len = log_data["original_shape"]
    crop_num = log_data.get("num_crops", crop_num)
    row_len = log_data.get("row_crop_size", row_len)
    col_len = log_data.get("col_crop_size", col_len)

    slice_start_indices, slice_end_indices = log_data["slice_start_indices"], log_data["slice_end_indices"]
    num_slices, fov_names = log_data["num_slices"], log_data["fov_names"]

    stitched_slices = np.zeros((fov_len, stack_len, crop_num, 1, row_len, col_len, 1))

    # loop slice indices to generate sliced data
    for i in range(num_slices - 1):
        stitched_slices[:, slice_start_indices[i]:slice_end_indices[i], :, 0, ...] = slice_stack[:, :, :, i, ...]

    # last slice, only index into stack the amount two indices are separated
    last_idx = num_slices - 1
    slice_len = slice_end_indices[last_idx] - slice_start_indices[last_idx]
    stitched_slices[:, slice_start_indices[last_idx]:slice_end_indices[last_idx], :, 0, ...] = slice_stack[:, :slice_len, :, last_idx, ...]

    stitched_xr = xr.DataArray(stitched_slices,
                               coords=[fov_names, range(stack_len), range(crop_num), range(1), range(row_len),
                                       range(col_len), ["segmentation_label"]],
                               dims=["fovs", "stacks", "crops", "slices", "rows", "cols", "channels"])
    return stitched_xr


def reconstruct_slice_data(save_dir, verbose=True):
    """High level function to put pieces of a slice back together
    Inputs
        save_dir: full path to directory where slice pieces are stored

    Outputs
        stitched_xr: xarray of [fovs, slices, rows, cols, segmentation_label] containing stitched labels"""

    if not os.path.isdir(save_dir):
        raise FileNotFoundError("slice directory does not exist")

    json_file_path = os.path.join(save_dir, "log_data.json")
    if not os.path.exists(json_file_path):
        raise FileNotFoundError("json file does not exist")

    with open(json_file_path) as json_file:
        slice_log_data = json.load(json_file)

    slice_stack = load_npzs(save_dir, slice_log_data, verbose=verbose)

    stitched_xr = stitch_slices(slice_stack, slice_log_data)

    return stitched_xr
