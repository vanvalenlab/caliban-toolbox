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

import os

import numpy as np
import xarray as xr


def find_sparse_images(labeled_data, cutoff=100):
    """Gets coordinates of images that have very few cells
    Args:
        labeled_data: predictions used for counting number of cells
        cutoff: minimum number of cells per image
    Returns:
        numpy.array: index used to remove sparse images
    """

    unique_counts = []
    for img in range(labeled_data.shape[0]):
        unique_counts.append(len(np.unique(labeled_data[img])) - 1)

    unique_counts = np.array(unique_counts)
    idx = unique_counts > cutoff

    return idx


def save_stitched_npzs(stitched_channels, stitched_labels, save_dir):
    """Takes corrected labels and channels and saves to NPZ for caliban round 2 checking
    Args:
        stitched_channels: original channel data
        stitched_labels: stitched labels
    """

    for i in range(stitched_channels.shape[0]):
        X = stitched_channels[i:(i + 1), ...]
        y = stitched_labels[i:(i + 1), ...]
        save_path = os.path.join(save_dir, stitched_labels.fovs.values[i] + '.npz')

        np.savez(save_path, X=X, y=y)


def process_stitched_data(base_dir):
    """Takes stitched output and creates folder of NPZs for review
    Args:
        base_dir: directory to read from
    """

    stitched_labels = xr.load_dataarray(os.path.join(base_dir, 'output', 'stitched_labels.xr'))
    channel_data = xr.load_dataarray(os.path.join(base_dir, 'channel_data.xr'))

    correction_folder = os.path.join(base_dir, 'ready_to_correct')
    os.makedirs(correction_folder)

    save_stitched_npzs(stitched_channels=channel_data, stitched_labels=stitched_labels,
                       save_dir=correction_folder)
