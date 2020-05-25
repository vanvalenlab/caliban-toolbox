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

import numpy as np

from deepcell_toolbox.utils import resize, tile_image


def combine_npz_files(npz_list, resize_ratios, stride_ratio=1, final_size=(256, 256)):
    """Take a series of NPZ files and combine together into single training NPZ

    Args:
        npz_list: list of NPZ files to combine
        resize_ratios: ratio used to resize each NPZ if data is of different resolutions
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
            # TODO: Modify tile_image to not produce overlapping crops
            current_x, _ = tile_image(image=current_x, model_input_shape=final_size,
                                      stride_ratio=stride_ratio)

            current_y, _ = tile_image(image=current_y, model_input_shape=final_size,
                                      stride_ratio=stride_ratio)

        combined_x.append(current_x)
        combined_y.append(current_y)

    combined_x = np.concatenate(combined_x, axis=0)
    combined_y = np.concatenate(combined_y, axis=0)

    return combined_x, combined_y
