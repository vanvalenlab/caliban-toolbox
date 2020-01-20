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
import copy
from skimage.segmentation import relabel_sequential


def combine_npz(npz_dir):
    """Reads all of the npzs in a directory, and aggregates them into a single stack

    Inputs:
        npz_dir: path to directory with npz files

    Outputs:
        stack: combined array of all labeled images"""

    # get all npz files
    files = os.listdir(npz_dir)
    files = [file for file in files if ".npz" in file]
    files.sort()
    print(files)

    # load first npz to get dimensions, create stack to hold all npz
    test_npz = np.load(os.path.join(npz_dir, files[0]))["y"]
    stack_num = len(files)
    stack = np.zeros((stack_num,) + test_npz.shape[:-1] + (1, ))

    # loop through all npz files, load into appropriate position in stack
    for idx, file in enumerate(files):
        npz = np.load(os.path.join(npz_dir, file))
        stack[idx, ...] = npz["y"]

    return stack


def stitch_crops(stack, padded_img_shape, row_starts, row_ends, col_starts, col_ends):
    """Takes a stack of annotated labels and stitches them together into a single image

    Inputs:
        y_stack: stack of annotations

    Outputs:
        stiched_image: stitched labels image"""

    # Initialize image
    stitched_image = np.zeros(padded_img_shape)

    crop_counter = 0

    for row in range(len(row_starts)):
        for col in range(len(col_starts)):

            # get crop and increment values to ensure unique labels across image
            crop = stack[crop_counter, ...]
            lowest_allowed_val = np.amax(stitched_image)
            crop = np.where(crop == 0, crop, crop + lowest_allowed_val)

            # get ids of cells in current crop
            potential_overlap_cells = np.unique(crop)
            potential_overlap_cells = potential_overlap_cells[np.nonzero(potential_overlap_cells)]

            # get values of stitched image at location where crop will be placed
            stitched_crop = stitched_image[row_starts[row]:row_ends[row], col_starts[col]:col_ends[col]]

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
            stitched_image[row_starts[row]:row_ends[row], col_starts[col]:col_ends[col]] = combined_crop

            crop_counter += 1

    # relabel image so that all cell_ids are present
    relabeled_stitch, _, _ = relabel_sequential(stitched_image)

    return relabeled_stitch


