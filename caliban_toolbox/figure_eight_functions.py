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
import json
import requests
import os
import stat
import zipfile
import pandas as pd
import urllib
import re

import getpass
from urllib.parse import urlencode

from caliban_toolbox.aws_functions import aws_upload_files, aws_copy_files, aws_download_files
from caliban_toolbox import crowdsource
from caliban_toolbox.utils.misc_utils import list_npzs_folder


def copy_job(job_id, key):
    """Helper function to create a Figure 8 job based on existing job.

    Args:
        job_id: ID number of job to copy instructions and settings from when creating new job
        key: API key to access Figure 8 account

    Returns:
        int: ID number of job created
    """

    url = 'https://api.appen.com/v1/jobs/{}/copy.json?'.format(str(job_id))
    API_key = {'key': key}

    new_job = requests.get(url, params=API_key)
    if new_job.status_code != 200:
        raise ValueError('copy_job not successful. Status code: '.format(new_job.status_code))

    new_job_id = new_job.json()['id']

    return new_job_id


def rename_job(job_id, key, name):
    """Helper function to create a Figure 8 job based on existing job.
        Args:
            job_id: ID number of job to rename
            key: API key to access Figure 8 account
            name: new name for job
    """

    payload = {
        'key': key,
        'job': {'title': name}
    }
    url = 'https://api.appen.com/v1/jobs/{}.json'.format(job_id)
    response = requests.put(url, json=payload)


def create_upload_log(base_dir, stage, aws_folder, filenames, filepaths, job_id, log_name,
                      pixel_only=False, label_only=False, rgb_mode=False):
    """Generates a csv log of parameters used during job creation for subsequent use in pipeline.

    Args:
        base_dir: full path to directory where job results will be stored
        stage: specifies stage in pipeline for jobs requiring multiple rounds of annotation
        aws_folder: folder in aws bucket where files be stored
        filenames: list of all files to be uploaded
        filepaths: list of complete urls to images in Amazon S3 bucket
        job_id: internal Figure Eight id for job
        log_name: name for log file
        pixel_only: flag specifying whether annotators will be restricted to pixel edit mode
        label_only: flag specifying whether annotators will be restricted to label edit mode
        rgb_mode: flag specifying whether annotators will view images in RGB mode
    """

    data = {'project_url': filepaths,
            'filename': filenames,
            'stage': stage,
            'aws_folder': aws_folder,
            'job_id': job_id,
            'pixel_only': pixel_only,
            'label_only': label_only,
            'rgb_mode': rgb_mode}
    dataframe = pd.DataFrame(data=data, index=range(len(filepaths)))

    # create file location, name file
    log_dir = os.path.join(base_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

        # add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(log_dir, mode)

    # save csv file
    dataframe.to_csv(os.path.join(log_dir, log_name), index=False)


def upload_log_file(log_file, job_id, key):
    """Upload log file to populate a job for Figure8

    Args:
        log_file: file specifying paths to NPZs included in this job
        job_id: ID number of job to upload data to
        key: API key to access Figure 8 account
    """

    # format url with appropriate arguments
    url = 'https://api.appen.com/v1/jobs/{}/upload.json?{}'
    url_dict = {'key': key, 'force': True}
    url_encoded_dict = urllib.parse.urlencode(url_dict)
    url = url.format(job_id, url_encoded_dict)

    headers = {'Content-Type': 'text/csv'}
    add_data = requests.put(url, data=log_file, headers=headers)

    if add_data.status_code != 200:
        raise ValueError('Upload_data not successful. Status code: '.format(add_data.status_code))
    else:
        print('Data successfully uploaded to Figure Eight.')
        return add_data.status_code


def create_figure_eight_job(base_dir, job_id_to_copy, aws_folder, stage, job_name=None,
                            rgb_mode=False, label_only=False, pixel_only=False):
    """Create a Figure 8 job and upload data to it. New job ID printed out for convenience.

    Args:
        base_dir: full path to job directory
        job_id_to_copy: ID number of Figure 8 job to use as template for new job
        aws_folder: folder in aws bucket where files be stored
        stage: specifies stage in pipeline for jobs requiring multiple rounds of annotation
        job_name: name for new job
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

    key = str(getpass.getpass("Figure eight api key? "))

    # copy job without data
    new_job_id = copy_job(job_id_to_copy, key)
    print('New job ID is: ' + str(new_job_id))

    # set name of new job
    if job_name is None:
        print("Job name not supplied, copying name from template job")
    else:
        rename_job(new_job_id, key, job_name)

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
    create_upload_log(base_dir=base_dir, stage=stage, aws_folder=aws_folder,
                      filenames=npzs, filepaths=url_paths, job_id=new_job_id,
                      pixel_only=pixel_only, rgb_mode=rgb_mode, label_only=label_only,
                      log_name=log_name)

    log_path = open(os.path.join(base_dir, 'logs', log_name), 'r')
    log_file = log_path.read()

    # upload log file
    status_code = upload_log_file(log_file, new_job_id, key)

    return status_code


def transfer_figure_eight_job(base_dir, job_id_to_copy, new_stage, job_name,
                              rgb_mode=False, label_only=False, pixel_only=False):
    """Create a Figure 8 job based on the output of a previous Figure8 job

    Args:
        base_dir: full path to job directory
        job_id_to_copy: ID number of Figure 8 job to use as template for new job
        new_stage: specifies new_stage for subsequent job
        job_name: name for next job
        pixel_only: flag specifying whether annotators will be restricted to pixel edit mode
        label_only: flag specifying whether annotators will be restricted to label edit mode
        rgb_mode: flag specifying whether annotators will view images in RGB mode
    """

    key = str(getpass("Figure eight api key?"))

    # copy job without data
    new_job_id = copy_job(job_id_to_copy, key)
    print('New job ID is: ' + str(new_job_id))

    # set name of new job
    rename_job(new_job_id, key, job_name)

    # get info from previous stage
    log_dir = os.path.join(base_dir, 'logs')
    previous_log_file = crowdsource.get_latest_log_file(log_dir)
    previous_log = pd.read_csv(os.path.join(log_dir, previous_log_file))
    filenames = previous_log['filename']
    previous_stage = previous_log['stage'][0]
    aws_folder = previous_log['aws_folder'][0]

    current_bucket = os.path.join(aws_folder, previous_stage)
    next_bucket = os.path.join(aws_folder, new_stage)

    # transfer files to new stage
    aws_copy_files(current_folder=current_bucket, next_folder=next_bucket,
                   filenames=filenames)

    new_log_name = crowdsource._create_next_log_name(previous_log_file, new_stage)

    # TODO: Decide if this should be handled by a separate function that is specific to transfer?
    _, _, filepaths, _ = crowdsource.create_job_urls(crop_dir=os.path.join(base_dir, 'crop_dir'),
                                                     aws_folder=aws_folder, stage=new_stage,
                                                     pixel_only=pixel_only, label_only=label_only,
                                                     rgb_mode=rgb_mode)

    # Generate log file for current job
    create_upload_log(base_dir=base_dir, stage=new_stage, aws_folder=aws_folder,
                      filenames=filenames, filepaths=filepaths, job_id=new_job_id,
                      pixel_only=pixel_only, rgb_mode=rgb_mode, label_only=label_only,
                      log_name=new_log_name)

    log_path = open(os.path.join(base_dir, 'logs', new_log_name), 'r')
    log_file = log_path.read()

    # upload log file
    upload_log_file(log_file, new_job_id, key)

    return log_file


def download_report(job_id, log_dir):
    """Download job report from Figure 8

    Args:
        job_id: Figure 8 job id
        log_dir: full path to log_dir where report will be saved
    """

    if not os.path.isdir(log_dir):
        print('Log directory does not exist: have you uploaded this job to Figure 8?')
        os.makedirs(log_dir)

        # add folder modification permissions to deal with files from file explorer
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(log_dir, mode)

    save_path = os.path.join(log_dir, 'job_report.zip')

    # password prompt for api info
    key = str(getpass.getpass("Please enter your Figure Eight API key:"))

    # construct url
    url = "https://api.appen.com/v1/jobs/{}.csv?".format(job_id)

    params = {"type": 'full', "key": key}

    # make http request: python requests handles redirects
    csv_request = requests.get(url, params=params, allow_redirects=True)
    open(save_path, 'wb').write(csv_request.content)
    print('Report saved to folder')


def unzip_report(log_dir):
    """Unzips .csv file and renames it appropriately

    Args:
        log_dir: full path to log_dir for saving zip
    """

    # Extract zip
    zip_path = os.path.join(log_dir, 'job_report.zip')
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        default_name = zip_ref.namelist()[0]  # get filename so can rename later
        zip_ref.extractall(log_dir)

    # rename from Figure 8 default
    default_name_path = os.path.join(log_dir, default_name)  # should only be one file in zip
    new_name_path = os.path.join(log_dir, 'job_report.csv')
    os.rename(default_name_path, new_name_path)


def download_figure_eight_output(base_dir):
    """Gets annotated files from a Figure 8 job

    Args:
        base_dir: directory containing relevant job files

    Returns:
        list: file names of NPZs not found in AWS bucket
    """

    # get information from job creation
    log_dir = os.path.join(base_dir, 'logs')
    latest_log = crowdsource.get_latest_log_file(log_dir)
    log_file = pd.read_csv(os.path.join(log_dir, latest_log))
    job_id = log_file['job_id'][0]

    # download Figure 8 report
    download_report(job_id=job_id, log_dir=log_dir)
    unzip_report(log_dir=log_dir)

    # download annotations from aws
    output_dir = os.path.join(base_dir, 'output')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    missing = aws_download_files(log_file, output_dir)

    return missing
