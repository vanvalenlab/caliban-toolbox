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

import pandas as pd

from caliban_toolbox import crowdsource
from caliban_toolbox.utils.misc_utils import list_npzs_folder
from caliban_toolbox.aws_functions import aws_upload_files, aws_download_files


def create_anolytics_job(base_dir, aws_folder, stage, rgb_mode=False, label_only=False,
                         pixel_only=False):
    """Create a log file and upload NPZs to aws for an anolytics job.

    Args:
        base_dir: full path to job directory
        aws_folder: folder in aws bucket where files be stored
        stage: specifies stage in pipeline for jobs requiring multiple rounds of annotation
        pixel_only: flag specifying whether annotators will be restricted to pixel edit mode
        label_only: flag specifying whether annotators will be restricted to label edit mode
        rgb_mode: flag specifying whether annotators will view images in RGB mode

    Raises:
        ValueError: If invalid base_dir supplied
        ValueError: If no crop directory found within base_dir
        ValueError: If no NPZs found in crop directory
    """

    if not os.path.isdir(base_dir):
        raise ValueError('Invalid directory name')

    upload_folder = os.path.join(base_dir, 'crop_dir')

    if not os.path.isdir(upload_folder):
        raise ValueError('No crop directory found within base directory')

    if len(list_npzs_folder(upload_folder)) == 0:
        raise ValueError('No NPZs found in crop dir')

    # get relevant paths
    npz_paths, npz_keys, url_paths, npzs = crowdsource.create_job_urls(crop_dir=upload_folder,
                                                                       aws_folder=aws_folder,
                                                                       stage=stage,
                                                                       pixel_only=pixel_only,
                                                                       label_only=label_only,
                                                                       rgb_mode=rgb_mode)

    # upload files to AWS bucket
    aws_upload_files(local_paths=npz_paths, aws_paths=npz_keys)

    log_name = 'stage_0_{}_upload_log.csv'.format(stage)

    # Generate log file for current job
    crowdsource.create_upload_log(base_dir=base_dir, stage=stage, aws_folder=aws_folder,
                                  filenames=npzs, filepaths=url_paths,
                                  pixel_only=pixel_only, rgb_mode=rgb_mode, label_only=label_only,
                                  log_name=log_name, separate_urls=True)


def download_anolytics_output(base_dir):
    """Gets annotated files from an anolytics job

    Args:
        base_dir: directory containing relevant job files

    Returns:
        list: file names of NPZs not found in AWS bucket
    """

    # get information from job creation
    log_dir = os.path.join(base_dir, 'logs')
    latest_log = crowdsource.get_latest_log_file(log_dir)
    log_file = pd.read_csv(os.path.join(log_dir, latest_log))

    # download annotations from aws
    output_dir = os.path.join(base_dir, 'output')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    missing = aws_download_files(log_file, output_dir)

    return missing
