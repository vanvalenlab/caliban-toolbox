# Copyright 2016-2020 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
import requests
import os
import stat
import zipfile
import pandas as pd

from getpass import getpass
from caliban_toolbox.log_file import create_upload_log
from caliban_toolbox.aws_functions import aws_upload_files, aws_download_files


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


def upload_data(csv_path, job_id, key):
    """Add data to an existing Figure 8 job by uploading a CSV file

    Args:
        csv_path: full path to csv
        job_id: ID number of job to upload data to
        key: API key to access Figure 8 account
    """

    url = "https://api.appen.com/v1/jobs/{job_id}/upload.json?key={api_key}&force=true"
    url = url.replace('{job_id}', str(job_id))
    url = url.replace('{api_key}', key)

    csv_file = open(csv_path, 'r')
    csv_data = csv_file.read()

    headers = {"Content-Type": "text/csv"}

    add_data = requests.put(url, data=csv_data, headers=headers)
    if add_data.status_code != 200:
        print("Upload_data not successful. Status code: ", add_data.status_code)
    else:
        print("Data successfully uploaded to Figure Eight.")


def create_figure_eight_job(base_dir, job_id_to_copy, aws_folder, stage,
                            rgb_mode=False, label_only=False, pixel_only=False):
    """Create a Figure 8 job and upload data to it. New job ID printed out for convenience.
    Args:
        base_dir: full path to directory that contains CSV files
        job_id_to_copy: ID number of Figure 8 job to use as template for new job
        aws_folder: folder in aws bucket where files be stored
        stage: specifies stage in pipeline for jobs requiring multiple rounds of annotation
        pixel_only: flag specifying whether annotators will be restricted to pixel edit mode
        label_only: flag specifying whether annotators will be restricted to label edit mode
        rgb_mode: flag specifying whether annotators will view images in RGB mode
    """

    key = str(getpass("Figure eight api key? "))

    # copy job without data
    new_job_id = copy_job(job_id_to_copy, key)
    if new_job_id == -1:
        return
    print('New job ID is: ' + str(new_job_id))

    # upload files to AWS bucket
    upload_folder = os.path.join(base_dir, 'crop_dir')
    filenames, filepaths = aws_upload_files(aws_folder=aws_folder, stage=stage,
                                            upload_folder=upload_folder, pixel_only=pixel_only,
                                            rgb_mode=rgb_mode, label_only=label_only)

    # Generate log file for current job
    create_upload_log(base_dir=base_dir, stage=stage, aws_folder=aws_folder,
                      filenames=filenames, filepaths=filepaths, job_id=new_job_id,
                      pixel_only=pixel_only, rgb_mode=rgb_mode, label_only=label_only)

    # upload NPZs using log file
    upload_data(os.path.join(base_dir, 'logs/upload_log.csv'), new_job_id, key)


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
    log_file = pd.read_csv(os.path.join(base_dir, 'logs/upload_log.csv'))
    job_id = log_file['job_id'][0]

    # download Figure 8 report
    log_dir = os.path.join(base_dir, 'logs')
    download_report(job_id=job_id, log_dir=log_dir)
    unzip_report(log_dir=log_dir)

    # download annotations from aws
    output_dir = os.path.join(base_dir, 'output')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    upload_log = pd.read_csv(os.path.join(base_dir, 'logs/upload_log.csv'))
    aws_download_files(upload_log, output_dir)
