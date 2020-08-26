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

import pandas as pd

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


def create_upload_log(base_dir, stage, aws_folder, filenames, filepaths, log_name, job_id=None,
                      pixel_only=False, label_only=False, rgb_mode=False, separate_urls=False):
    """Generates a csv log of parameters used during job creation for subsequent use in pipeline.

    Args:
        base_dir: full path to directory where job results will be stored
        stage: specifies stage in pipeline for jobs requiring multiple rounds of annotation
        aws_folder: folder in aws bucket where files be stored
        filenames: list of all files to be uploaded
        filepaths: list of complete urls to images in Amazon S3 bucket
        log_name: name for log file
        job_id: job_id for Figure8 jobs
        pixel_only: flag specifying whether annotators will be restricted to pixel edit mode
        label_only: flag specifying whether annotators will be restricted to label edit mode
        rgb_mode: flag specifying whether annotators will view images in RGB mode
        separate_urls: if True save a separate CSV containing just the URLs
    """

    data = {'project_url': filepaths,
            'filename': filenames,
            'stage': stage,
            'aws_folder': aws_folder,
            'pixel_only': pixel_only,
            'label_only': label_only,
            'rgb_mode': rgb_mode}
    dataframe = pd.DataFrame(data=data, index=range(len(filepaths)))

    if job_id is not None:
        dataframe['job_id'] = job_id

    # create file location, name file
    log_dir = os.path.join(base_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # save csv file
    dataframe.to_csv(os.path.join(log_dir, log_name), index=False)

    # create csv containing only URLs
    if separate_urls:
        url_df = pd.DataFrame({'project_url': filepaths})
        url_df.to_csv(os.path.join(log_dir, 'url_only_' + log_name))
