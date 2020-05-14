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
import pandas as pd
import xarray as xr

from caliban_toolbox import metadata
from caliban_toolbox.utils.pipeline_utils import get_job_folder_name


def create_experiment_folder(image_names, raw_metadata, base_dir):
    """Takes the output of the data loader and creates an experiment folder

    Args:
        image_names: names of images from current experiment
        raw_metadata: metadata file from raw ontology
        base_dir: directory where experiment folder will be created

    Returns:
        string: full path to newly created experiment folder
    """

    experiment_id = raw_metadata['EXPERIMENT_ID']
    experiment_folder = os.path.join(base_dir, 'experiment_{}'.format(experiment_id))
    os.makedirs(experiment_folder)

    # create metadata file
    exp_metadata = metadata.make_experiment_metadata_file(raw_metadata, image_names)

    # save metadata file
    exp_metadata.to_csv(os.path.join(experiment_folder, 'metadata.csv'))

    return experiment_folder


def create_job_folder(experiment_dir, metadata, fov_data, fov_names, fov_num):
    """Creates a folder to hold a single caliban job

    Args:
        experiment_dir: directory of relevant experiment
        fov_num: number of FOVs to include in job
    """

    # Create sequentially named job folder
    job_folder_path, job_name = get_job_folder_name(experiment_dir)
    os.makedirs(job_folder_path)

    available_fovs = metadata[metadata['status'] == 'awaiting_prediction']
    new_fov_names = available_fovs['image_name'][:fov_num].values

    metadata.loc[metadata['image_name'].isin(new_fov_names),
                 ['status', 'job_name']] = 'in_progress', job_name

    fov_idx = np.isin(fov_names, new_fov_names)

    new_fov_data = fov_data[fov_idx]

    np.savez(os.path.join(job_folder_path, 'raw_data.npz'), X=new_fov_data)
    metadata.to_csv(os.path.join(experiment_dir, 'metadata.csv'))


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
