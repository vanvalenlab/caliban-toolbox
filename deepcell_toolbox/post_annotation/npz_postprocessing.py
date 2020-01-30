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

from skimage.segmentation import relabel_sequential


def combine_npz(npz_dir, fov_names, row_crop_size, col_crop_size, num_row_crops, num_col_crops):
    """Reads all of the npzs in a directory, and aggregates them into a single stack

    Inputs:
        npz_dir: path to directory with npz files
        file_header: pattern to search for in NPZ files

    Outputs:
        stack: combined array of all labeled images"""

    stack = np.zeros((len(fov_names), num_col_crops*num_row_crops, row_crop_size, col_crop_size, 1))

    # loop through all npz files, load into appropriate position in stack
    for fov_idx, fov_name in enumerate(fov_names):
        crop_idx = 0
        for row in range(num_row_crops):
            for col in range(num_col_crops):
                npz_path = os.path.join(npz_dir, "{}_row_{}_col_{}.npz".format(fov_name, row, col))
                if os.path.exists(npz_path):
                    npz = np.load(npz_path)["y"]
                    stack[fov_idx, crop_idx, ...] = npz
                else:
                    # npz not generated, did not contain any labels, keep blank
                    print("could not find npz {}".format(npz_path))
                crop_idx += 1

    return stack


def stitch_crops(stack, padded_img_shape, row_starts, row_ends, col_starts, col_ends):
    """Takes a stack of annotated labels and stitches them together into a single image

    Inputs:
        stack: stack of crops to be stitched together
        padded_img_shape: shape of the original padded image
        row_starts: list of row indices for crop starts
        row_ends: list of row indices for crops ends
        col_starts: list of col indices for col starts
        col_ends: list of col indices for col ends

    Outputs:
        relabeled_stitch: stitched labels image, sequentially relabeled"""

    # Initialize image
    stitched_image = np.zeros(padded_img_shape)

    # loop through all crops in the stack
    for img in range(stack.shape[0]):
        crop_counter = 0
        for row in range(len(row_starts)):
            for col in range(len(col_starts)):

                # get crop and increment values to ensure unique labels across final image
                crop = stack[img, crop_counter, ...]
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
                    stitched_overlap_vals = stitched_overlap_vals[np.nonzero(stitched_overlap_vals)]

                    # if there are overlaps, determine which is greatest, and replace with that value
                    if len(stitched_overlap_vals) > 0:
                        max_overlap = stitched_overlap_vals[np.argmax(stitched_overlap_vals)]
                        crop[crop == cell] = max_overlap

                # combine the crop with the current values in the stitched image
                combined_crop = np.where(stitched_crop > 0, stitched_crop, crop)

                # use this combined crop to update the values of stitched image
                stitched_image[img, row_starts[row]:row_ends[row], col_starts[col]:col_ends[col]] = combined_crop

                crop_counter += 1

    # # relabel image so that all cell_ids are present
    # relabeled_stitch, _, _ = relabel_sequential(stitched_image)

    return stitched_image


def reconstruct_npz(npz_dir, original_npz):
    """High level function to combine npz crops together into a single stitched image

    Inputs:
        npz_dir: directory where cropped npz files are stored
        original_npz: path to original npz file to load the channel data from

    Outputs:
        None (saves stitched npz to folder)"""

    # unpack JSON data
    with open(os.path.join(npz_dir, "log_data.json")) as json_file:
        log_data = json.load(json_file)

    num_crops, row_start, row_end = log_data["num_crops"], log_data["row_start"], log_data["row_end"]
    col_start, col_end, padded_shape = log_data["col_start"], log_data["col_end"], log_data["padded_shape"]
    row_padding, col_padding, fov_names = log_data["row_padding"], log_data["col_padding"], log_data["fov_names"]

    # combine all npz crops into a single stack
    npz_stack = combine_npz(npz_dir=npz_dir, fov_names=fov_names, row_crop_size=row_end[0]-row_start[0],
                            col_crop_size=col_end[0]-col_start[0], num_row_crops=len(row_start),
                            num_col_crops=len(col_start))

    # stitch crops together into single contiguous image
    stitched_images = stitch_crops(stack=npz_stack, padded_img_shape=padded_shape, row_starts=row_start,
                                   row_ends=row_end, col_starts=col_start, col_ends=col_end)

    # crop image down to original size
    stitched_images = stitched_images[:, 0:(-row_padding), 0:(-col_padding), :]

    # combine newly generated stitched labels with original channels data
    original_npz = np.load(original_npz)

    np.savez(os.path.join(npz_dir, "stitched_npz.npz"), X=original_npz["X"], y=stitched_images)
