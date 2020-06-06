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

from getpass import getpass
from urllib.parse import urlencode

from caliban_toolbox.log_file import create_upload_log
from caliban_toolbox.aws_functions import aws_upload_files, aws_transfer_files, aws_download_files
from caliban_toolbox.utils.misc_utils import list_npzs_folder


def _format_url(aws_folder, stage, npz, url_encoded_dict):
    base_url = 'https://caliban.deepcell.org/caliban-input__caliban-output__{}__{}__{}?{}'
    formatted_url = base_url.format(aws_folder, stage, npz, url_encoded_dict)

    return formatted_url


def _create_next_log_name(previous_log_name, stage):
    stage_num = previous_log_name.split('_')[1]
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

    # TODO: think about better way to structure than than many lists
    return npz_paths, npz_keys, url_paths, npzs_to_upload


def copy_job(job_id, key):
    """Helper function to create a Figure 8 job based on existing job.

    Args:
        job_id: ID number of job to copy instructions and settings from when creating new job
        key: API key to access Figure 8 account

    Returns:
        int: ID number of job created
    """

    url = 'https://api.appen.com/v1/jobs/{}/copy.json?'.format(str(job_id))
    API_key = {"key": key}

    new_job = requests.get(url, params=API_key)
    if new_job.status_code != 200:
        print("copy_job not successful. Status code: ", new_job.status_code)
    new_job_id = new_job.json()['id']

    return new_job_id


def rename_job(job_id, key, name):
    """Helper function to create a Figure 8 job based on existing job.

    Args:
        job_id: ID number of job to rename
        key: API key to access Figure 8 account
        name: new name for job
    """

    headers = {'content-type': 'application/json'}
    payload = {
        'key': key,
        'job': {
            'title': name
        }}
    response = requests.put(
        'https://api.figure-eight.com/v1/jobs/{}.json'.format(job_id), data=json.dumps(payload),
        headers=headers)


def upload_log_file(log_file, job_id, key):
    """Upload log file to populate a job for Figure8

    Args:
        log_file: file specifying paths to NPZs included in this job
        job_id: ID number of job to upload data to
        key: API key to access Figure 8 account
    """

    # format url with appropriate arguments
    url = "https://api.appen.com/v1/jobs/{}/upload.json?{}"
    url_dict = {'key': key, 'force': True}
    url_encoded_dict = urllib.parse.urlencode(url_dict)
    url = url.format(job_id, url_encoded_dict)

    headers = {"Content-Type": "text/csv"}
    add_data = requests.put(url, data=log_file, headers=headers)

    if add_data.status_code != 200:
        print("Upload_data not successful. Status code: ", add_data.status_code)
    else:
        print("Data successfully uploaded to Figure Eight.")


def create_figure_eight_job(base_dir, job_id_to_copy, job_name, aws_folder, stage,
                            rgb_mode=False, label_only=False, pixel_only=False):
    """Create a Figure 8 job and upload data to it. New job ID printed out for convenience.

    Args:
        base_dir: full path to job directory
        job_id_to_copy: ID number of Figure 8 job to use as template for new job
        job_name: name for new job
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

    key = str(getpass("Figure eight api key? "))

    # copy job without data
    new_job_id = copy_job(job_id_to_copy, key)
    print('New job ID is: ' + str(new_job_id))

    # set name of new job
    rename_job(new_job_id, key, job_name)
    
    # get relevant paths
    npz_paths, npz_keys, url_paths, npzs = create_job_urls(crop_dir=upload_folder,
                                                           aws_folder=aws_folder,
                                                           stage=stage, pixel_only=pixel_only,
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
    upload_log_file(log_file, new_job_id, key)


def transfer_figure_eight_job(base_dir, job_id_to_copy, new_stage,
                              rgb_mode=False, label_only=False, pixel_only=False):
    """Create a Figure 8 job based on the output of a previous Figure8 job

    Args:
        base_dir: full path to job directory
        job_id_to_copy: ID number of Figure 8 job to use as template for new job
        new_stage: specifies new_stage for subsequent job
        pixel_only: flag specifying whether annotators will be restricted to pixel edit mode
        label_only: flag specifying whether annotators will be restricted to label edit mode
        rgb_mode: flag specifying whether annotators will view images in RGB mode
    """

    key = str(getpass("Figure eight api key?"))

    # copy job without data
    new_job_id = copy_job(job_id_to_copy, key)
    print('New job ID is: ' + str(new_job_id))

    # get info from previous stage
    log_dir = os.path.join(base_dir, 'logs')
    previous_log_file = get_latest_log_file(log_dir)
    previous_log = pd.read_csv(os.path.join(log_dir, previous_log_file))
    filenames = previous_log['filename']
    previous_stage = previous_log['stage'][0]
    aws_folder = previous_log['aws_folder'][0]

    # transfer files to new stage
    filepaths = aws_transfer_files(aws_folder=aws_folder, completed_stage=previous_stage,
                                   new_stage=new_stage, files_to_transfer=filenames,
                                   pixel_only=pixel_only,
                                   rgb_mode=rgb_mode, label_only=label_only)

    new_log_name = _create_next_log_name(previous_log_file, new_stage)

    # Generate log file for current job
    create_upload_log(base_dir=base_dir, stage=new_stage, aws_folder=aws_folder,
                      filenames=filenames, filepaths=filepaths, job_id=new_job_id,
                      pixel_only=pixel_only, rgb_mode=rgb_mode, label_only=label_only,
                      log_name=new_log_name)

    # upload log file
    upload_log_file(os.path.join(log_dir, new_log_name), new_job_id, key)


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
    key = str(getpass("Please enter your Figure Eight API key:"))

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
    """

    # get information from job creation
    log_dir = os.path.join(base_dir, 'logs')
    latest_log = get_latest_log_file(log_dir)
    log_file = pd.read_csv(os.path.join(log_dir, latest_log))
    job_id = log_file['job_id'][0]

    # download Figure 8 report
    download_report(job_id=job_id, log_dir=log_dir)
    unzip_report(log_dir=log_dir)

    # download annotations from aws
    output_dir = os.path.join(base_dir, 'output')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    aws_download_files(log_file, output_dir)
