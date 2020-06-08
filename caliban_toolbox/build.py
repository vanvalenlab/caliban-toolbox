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

import math

import numpy as np

from deepcell_toolbox.utils import resize, tile_image


def pad_image_stack(images, crop_size):
    """Pads an an array of images so that it is divisible by the specified crop_size

    Args:
        images: array of images to be cropped
        crop_size: tuple specifying crop size

    Returns:
        np.array: padded image stack
    """

    row_len, col_len = images.shape[1:3]
    row_crop, col_crop = crop_size
    row_num = math.ceil(row_len / crop_size[0])
    col_num = math.ceil(col_len / crop_size[1])

    new_row_len = row_num * row_crop
    new_col_len = col_num * col_crop

    if new_row_len == row_len and new_col_len == col_len:
        # don't need to pad
        return images
    else:
        new_images = np.zeros((images.shape[0], new_row_len, new_col_len, images.shape[3]))
        new_images[:, :row_len, :col_len, :] = images
        return new_images


def pad_image_edge_crops(images, crop_size):
    """Add padding to an image to enable additional edge crops to be generated

    Args:
        images: array of images to crop
        crop_size: size of crops that will be produced

    Returns:
        np.array: padded image
    """
    row_offset, col_offset = int(crop_size[0] / 2), int(crop_size[1] / 2)

    padded_images = np.zeros((images.shape[0], images.shape[1] + crop_size[0],
                              images.shape[2] + crop_size[1], images.shape[3]))

    padded_images[:, row_offset:-row_offset, col_offset:-col_offset, :] = images

    return padded_images


def combine_npz_files(npz_list, resize_ratios, extra_edge_padding=False,
                      stride_ratio=1, final_size=(256, 256)):
    """Take a series of NPZ files and combine together into single training NPZ

    Args:
        npz_list: list of NPZ files to combine
        resize_ratios: ratio used to resize each NPZ if data is of different resolutions
        extra_edge_padding: if True, adds padding equal to half of the crop size on either edge
            of the image to increase the number of crops produced from the original image
        stride_ratio: amount of overlap between crops (1 is no overlap, 0.5 is half crop size)
        final_size: size of the final crops to be produced
    Returns:
        np.array: array containing resized and cropped data from all input NPZs
    Raises:
        ValueError: If resize ratios are not integers
    """

    combined_x = []
    combined_y = []

    for idx, npz in enumerate(npz_list):
        current_x = npz['X']
        current_y = npz['y']

        # resize if needed
        current_resize = resize_ratios[idx]
        if current_resize != 1:
            new_shape = (int(current_x.shape[1] * current_resize),
                         int(current_x.shape[2] * current_resize))

            current_x = resize(data=current_x, shape=new_shape)
            current_y = resize(data=current_y, shape=new_shape, labeled_image=True)

        # crop if needed
        if current_x.shape[1:3] != final_size:

            # pad image so that crops divide evenly
            current_x = pad_image_stack(images=current_x, crop_size=final_size)
            current_y = pad_image_stack(images=current_y, crop_size=final_size)

            if extra_edge_padding:
                # create buffer on either edge to increase number of crops
                current_x = pad_image_edge_crops(images=current_x, crop_size=final_size)
                current_y = pad_image_edge_crops(images=current_y, crop_size=final_size)

            current_x, _ = tile_image(image=current_x, model_input_shape=final_size,
                                      stride_ratio=stride_ratio)
            current_y, _ = tile_image(image=current_y, model_input_shape=final_size,
                                      stride_ratio=stride_ratio)

        combined_x.append(current_x)
        combined_y.append(current_y)

    combined_x = np.concatenate(combined_x, axis=0)
    combined_y = np.concatenate(combined_y, axis=0)

    return combined_x, combined_y
