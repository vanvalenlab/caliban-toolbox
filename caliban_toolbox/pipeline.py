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
import json

import numpy as np

from caliban_toolbox import metadata


def create_experiment_folder(data_loader_output, raw_metadata, base_dir):
    """Takes the output of the data loader and creates an experiment folder

    Args:
        data_loader_output: files from an experiment to be processed
        raw_metadata: metadata file from raw ontology
        base_dir: directory where experiment folder will be created
    """

    experiment_id = raw_metadata['EXPERIMENT_ID']
    experiment_folder = os.path.join(base_dir, experiment_id)
    os.makedirs(experiment_folder)

    # create metadata file
    exp_metadata = metadata.make_experiment_metadata_file(raw_metadata)
    exp_metadata['in_progress_fovs'] = data_loader_output.fovs

    # save metadata file
    with open(experiment_folder, 'w') as write_file:
        json.dump(exp_metadata, write_file)


def create_job_folder(experiment_dir, fov_num):
    """Creates a folder to hold a single caliban job

    Args:
        experiment_dir: directory of relevant experiment
        fov_num: number of FOVs to include in job
    """

    # Check if job folders already exist
    folder_name = get_next_folder_name(experiment_dir)
    os.makedirs(folder_name)

    with open(os.path.join(experiment_dir, 'metadata.json')) as json_file:
        exp_metadata = json.load(json_file)

    new_fovs = get_fovs_from_metadata(exp_metadata)

    job_metadata = metadata.make_job_metadata_file(exp_metadata, new_fovs)


def save_stitched_npzs(stitched_channels, stitched_labels, save_dir):
    """Takes corrected labels and channels and saves to NPZ for caliban round 2 checking

    Args:
        stitched_channels: original channel data
        stitched_labels: stitched labels
    """

    for i in range(stitched_channels.shape[0]):
        X = stitched_labels[i:(i + 1), ...]
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

