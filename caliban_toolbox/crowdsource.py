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
import re

from urllib.parse import urlencode

from caliban_toolbox.utils.misc_utils import list_npzs_folder


def _format_url(aws_folder, stage, npz, url_encoded_dict):
    base_url = 'https://caliban.deepcell.org/caliban-input__caliban-output__{}__{}__{}?{}'
    formatted_url = base_url.format(aws_folder, stage, npz, url_encoded_dict)

    return formatted_url


def _create_next_log_name(previous_log_name, stage):
    stage_num = int(previous_log_name.split('_')[1])
    new_log = 'stage_{}_{}_upload_log.csv'.format(stage_num + 1, stage)

    return new_log


def get_latest_log_file(log_dir):
    """Find the latest log file in the log directory

    Args:
        log_dir: full path to log directory

    Returns:
        string: name of the latest log file
    """
    files = os.listdir(log_dir)
    log_files = [file for file in files if 'upload_log.csv' in file]
    log_files.sort()

    return log_files[-1]


def create_job_urls(crop_dir, aws_folder, stage, pixel_only, label_only, rgb_mode):
    """Helper function to create relevant URLs for caliban log and AWS upload

    Args:
        crop_dir: full path to directory with the npz crops
        aws_folder: path for images to be stored in AWS
        stage: which stage of the correction process this job is for
        pixel_only: boolean flag to determine if only pixel mode is available
        label_only: boolean flag to determine if only label is available
        rgb_mode: boolean flag to determine if rgb mode will be enabled

    Returns:
        list: list of paths to local NPZs to be uploaded
        list: list of paths to desintation for NPZs
        list: list of URLs to supply to figure8 to to display crops
        list: list of NPZs that will be uploaded

    Raises:
        ValueError: If URLs are not valid
    """
    # TODO: check that URLS don't contain invalid character
    # load the images from specified folder but not the json log file
    npzs_to_upload = list_npzs_folder(crop_dir)

    # change slashes separating nested folders to underscores for URL generation
    subfolders = re.split('/', aws_folder)
    subfolders = '__'.join(subfolders)

    # create dictionary to hold boolean flags
    url_dict = {'pixel_only': pixel_only, 'label_only': label_only, 'rgb': rgb_mode}
    url_encoded_dict = urlencode(url_dict)

    # create path to npz, key to upload npz, and url path for figure8
    npz_paths, npz_keys, url_paths = [], [], []
    for npz in npzs_to_upload:
        npz_paths.append(os.path.join(crop_dir, npz))
        npz_keys.append(os.path.join(aws_folder, stage, npz))
        url_paths.append(_format_url(subfolders, stage, npz, url_encoded_dict))

    # TODO: think about better way to structure than many lists
    return npz_paths, npz_keys, url_paths, npzs_to_upload
