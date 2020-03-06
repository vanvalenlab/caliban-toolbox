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

from skimage.segmentation import relabel_sequential
from segmentation.utils import data_utils


def load_crops(crop_dir, fov_names, row_crop_size, col_crop_size, num_row_crops, num_col_crops, save_format):
    """Reads all of the cropped images from a directory, and aggregates them into a single stack

    Inputs:
        crop_dir: path to directory with cropped npz or xarray files
        fov_names: list of unique fov names in dataset
        row_crop_size: size of the crops in rows dimension
        col_crop_size: size of the crops in cols dimension
        num_row_crops: number of crops in rows dimension
        num_col_cops: number of crops in cols dimension
        save_format: format in which the crops were saved

    Outputs:
        stack: combined array of all labeled images"""

    stack = np.zeros((len(fov_names), num_col_crops*num_row_crops, row_crop_size, col_crop_size, 1))

    # loop through all npz files
    for fov_idx, fov_name in enumerate(fov_names):
        crop_idx = 0
        for row in range(num_row_crops):
            for col in range(num_col_crops):

                # if data was saved as npz, load X and y separately then combine
                if save_format == "npz":
                    npz_path = os.path.join(crop_dir, "{}_row_{}_col_{}_save_version_0.npz".format(fov_name, row, col))
                    if os.path.exists(npz_path):
                        temp_npz = np.load(npz_path)
                        stack[fov_idx:(fov_idx + 1), crop_idx, ...] = temp_npz["y"]
                    else:
                        # npz not generated, did not contain any labels, keep blank
                        print("could not find npz {}, skipping".format(npz_path))

                # if data was saved as xr, load both X and y simultaneously
                elif save_format == "xr":
                    xr_path = os.path.join(crop_dir, "{}_row_{}_col_{}.xr".format(fov_name, row, col))
                    if os.path.exists(xr_path):
                        temp_xr = xr.open_dataarray(xr_path)
                        stack[fov_idx:(fov_idx + 1), crop_idx, ...] = temp_xr[..., -1:]
                    else:
                        # npz not generated, did not contain any labels, keep blank
                        print("could not find xr {}, skipping".format(xr_path))
                crop_idx += 1

    return stack


def stitch_crops(stack, padded_img_shape, row_starts, row_ends, col_starts, col_ends, relabel=True):
    """Takes a stack of annotated labels and stitches them together into a single image

    Inputs:
        stack: stack of crops to be stitched together
        padded_img_shape: shape of the original padded image
        row_starts: list of row indices for crop starts
        row_ends: list of row indices for crops ends
        col_starts: list of col indices for col starts
        col_ends: list of col indices for col ends

    Outputs:
        stitched_image: stitched labels image, sequentially relabeled"""

    # Initialize image with single dimension for channels
    stitched_shape = list(padded_img_shape[:-1])
    stitched_shape = stitched_shape + [1]
    stitched_image = np.zeros(stitched_shape)

    # loop through all crops in the stack for each image
    for img in range(stack.shape[0]):
        crop_counter = 0
        for row in range(len(row_starts)):
            for col in range(len(col_starts)):

                # get current crop
                crop = stack[img, crop_counter, ...]

                # increment values to ensure unique labels across final image
                lowest_allowed_val = np.amax(stitched_image[img, ...])
                crop = np.where(crop == 0, crop, crop + lowest_allowed_val)

                # get ids of cells in current crop
                potential_overlap_cells = np.unique(crop)
                potential_overlap_cells = potential_overlap_cells[np.nonzero(potential_overlap_cells)]

                # get values of stitched image at location where crop will be placed
                stitched_crop = stitched_image[img, row_starts[row]:row_ends[row], col_starts[col]:col_ends[col], :]

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
                stitched_image[img, row_starts[row]:row_ends[row], col_starts[col]:col_ends[col]] = combined_crop

                crop_counter += 1

    # relabel images to remove skipped cell_ids
    if relabel:
        for img in range(stitched_image.shape[0]):
            stitched_image[img, ..., -1], _, _ = relabel_sequential(stitched_image[img, ..., -1])

    return stitched_image


def reconstruct_image_stack(crop_dir, save_format="xr", relabel=True):
    """High level function to combine crops together into a single stitched image

    Inputs:
        crop_dir: directory where cropped files are stored
        save_format: format that crops were saved in

    Outputs:
        None (saves stitched xarray to folder)"""

    # sanitize inputs
    if not os.path.isdir(crop_dir):
        raise ValueError("crop_dir not a valid directory: {}".format(crop_dir))

    if save_format not in ["xr", "npz"]:
        raise ValueError("save_format needs to be one of ['xr', 'npz'], got {}".format(save_format))

    # unpack JSON data
    with open(os.path.join(crop_dir, "log_data.json")) as json_file:
        log_data = json.load(json_file)

    row_start, row_end = log_data["row_start"], log_data["row_end"]
    col_start, col_end, padded_shape = log_data["col_start"], log_data["col_end"], log_data["padded_shape"]
    row_padding, col_padding, fov_names = log_data["row_padding"], log_data["col_padding"], log_data["fov_names"]
    chan_names = log_data["chan_names"]

    # combine all npz crops into a single stack
    crop_stack = load_crops(crop_dir=crop_dir, fov_names=fov_names, row_crop_size=row_end[0]-row_start[0],
                               col_crop_size=col_end[0]-col_start[0], num_row_crops=len(row_start),
                               num_col_crops=len(col_start), save_format=save_format)

    # stitch crops together into single contiguous image
    stitched_images = stitch_crops(stack=crop_stack, padded_img_shape=padded_shape, row_starts=row_start,
                                   row_ends=row_end, col_starts=col_start, col_ends=col_end, relabel=relabel)

    # crop image down to original size
    stitched_images = stitched_images[:, 0:(-row_padding), 0:(-col_padding), :]

    stitched_xr = xr.DataArray(data=stitched_images,
                               coords=[fov_names, range(stitched_images.shape[1]), range(stitched_images.shape[2]),
                                       chan_names[-1:]], dims=["fovs", "rows", "cols", "channels"])

    stitched_xr.to_netcdf(os.path.join(crop_dir, "stitched_images.nc"))


def overlay_grid_lines(overlay_img, row_start, row_end, col_start, col_end):
    """Visualize the location of image crops on the original uncropped image to assess crop size

    Inputs
        overlay_img: original image [rows x cols] that crops will overlaid onto
        row_start: vector of start indices generated by crop_multichannel_data
        row_end: vector of end indices generated by crop_multichannel_data
        col_start: vector of start indices generated by crop_multichannel_data
        col_end: vector of end indices generated by crop_multichannel_data

    Outputs
        overlay_img: original image overlaid with the crop borders"""

    # get dimensions of the image
    row_len = overlay_img.shape[0]
    col_len = overlay_img.shape[1]

    # if first start position is 0, we won't plot it since it's on the image border
    if row_start[0] == 0:
        row_start = row_start[1:]
    if col_start[0] == 0:
        col_start = col_start[1:]

    # use distance between start index of crop 1 and end index of crop 0 determine overlap amount
    row_sep = row_end[0] - row_start[0]
    col_sep = col_end[0] - col_start[0]

    # generate a vector of alternating 0s and im_max for a dotted line
    val = np.max(overlay_img)
    dotted = [0, 0, 0, 0, val, val, val, val]
    side = np.max(overlay_img.shape)
    dotted = dotted * side

    # trim dotted vectors to be same length as respective image dimension
    row_dotted = np.expand_dims(np.array(dotted[:row_len]), axis=-1)
    col_dotted = np.expand_dims(np.array(dotted[:col_len]), axis=0)

    # expand the thickness to 3 pixel width for better visualization
    row_start = row_start + [x + 1 for x in row_start] + [x + 2 for x in row_start]
    row_end = [x + row_sep for x in row_start]

    # define the location of each image to be halfway between the overlap boundary on either side
    row_middle = [int(x + (row_sep / 2)) for x in row_start]

    # same for columns
    col_start = col_start + [x + 1 for x in col_start] + [x + 2 for x in col_start]
    col_middle = [int(x + (col_sep / 2)) for x in col_start]
    col_end = [x + col_sep for x in col_start]

    # set the values of start and end indices to be dotted lines
    overlay_img[row_start, :] = col_dotted
    overlay_img[row_end, :] = col_dotted
    overlay_img[:, col_start] = row_dotted
    overlay_img[:, col_end] = row_dotted

    # set the values of the line delineating image crop to be constant value
    overlay_img[row_middle, :] = val
    overlay_img[:, col_middle] = val

    return overlay_img


def overlay_crop_overlap(img_crop, row_start, row_end, col_start, col_end):
    """Visualize the extent of overlap between adjacent crops by plotting overlap regions on an example crop

    Inputs
        img_crop: example crop generated by crop_multichannel_data over which overlap will be plotted
        row_start: vector of start indices generated by crop_multichannel_data
        row_end: vector of end indices generated by crop_multichannel_data
        col_start: vector of start indices generated by crop_multichannel_data
        col_end: vector of end indices generated by crop_multichannel_data

    Outputs
        img_crop: example crop with dotted lines superimposed on location of adjacent overlapping crops"""

    # get image dimensions
    row_len, col_len = img_crop.shape[0], img_crop.shape[1]

    # compute amount of overlap on each axis
    row_overlap = row_end[0] - row_start[1]
    col_overlap = col_end[0] - col_start[1]

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

