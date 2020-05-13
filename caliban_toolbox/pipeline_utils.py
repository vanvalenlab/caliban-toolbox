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


def get_job_folder_name(experiment_dir):
    """Identify the name for next sequentially named job folder

    Args:
        experiment_dir: full path to directory of current experiment

    Returns:
        string: full path to newly created job folder
        string: name of the job folder
    """

    files = os.listdir(experiment_dir)
    folders = [file for file in files if os.path.isdir(os.path.join(experiment_dir, file))]
    folders = [folder for folder in folders if 'caliban_job_' in folder]
    folders.sort()

    if len(folders) == 0:
        new_folder = 'caliban_job_0'
    else:
        latest_folder_num = folders[-1].split('caliban_job_')[1]
        new_folder = 'caliban_job_{}'.format(latest_folder_num)

    new_folder_path = os.path.join(experiment_dir, new_folder)

    return new_folder_path, new_folder

