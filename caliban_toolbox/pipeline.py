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

    # make folder for current experiment
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
        metadata: metadata file for experiment
        fov_data: image data for current experiment
        fov_names: names of FOVs in current experiment
        fov_num: number of FOVs to include in job
    """

    # Create sequentially named job folder
    job_folder_path, job_name = get_job_folder_name(experiment_dir)
    os.makedirs(job_folder_path)

    # get specified number of new FOVs from list of available FOVs
    available_fovs = metadata[metadata['status'] == 'awaiting_prediction']
    new_fov_names = available_fovs['image_name'][:fov_num].values

    # update the status of selected FOVs
    metadata.loc[metadata['image_name'].isin(new_fov_names),
                 ['status', 'job_folder']] = 'in_progress', job_name

    # get image data corresponding to selected FOVs
    fov_idx = np.isin(fov_names, new_fov_names)
    new_fov_data = fov_data[fov_idx]

    # save image data and metadata file
    np.savez(os.path.join(job_folder_path, 'raw_data.npz'), X=new_fov_data)
    metadata.to_csv(os.path.join(experiment_dir, 'metadata.csv'))
