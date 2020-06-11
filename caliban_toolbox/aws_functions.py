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
import sys
import threading
import re

import boto3
import botocore

from urllib.parse import urlencode
from getpass import getpass

from caliban_toolbox.utils.misc_utils import list_npzs_folder


# Taken from AWS Documentation
class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()


def connect_aws():
    AWS_ACCESS_KEY_ID = getpass('What is your AWS access key id? ')
    AWS_SECRET_ACCESS_KEY = getpass('What is your AWS secret access key id? ')

    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID,
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    print('Connected to AWS')
    s3 = session.client('s3')

    return s3


def aws_upload_files(local_paths, aws_paths):
    """Uploads files to AWS bucket for use in Figure 8

        Args:
            local_paths: list of paths to npz files
            aws_paths: list of paths for saving npz files in AWS
    """

    s3 = connect_aws()

    # upload images
    for i in range(len(local_paths)):
        s3.upload_file(local_paths[i], 'caliban-input', aws_paths[i],
                       Callback=ProgressPercentage(local_paths[i]),
                       ExtraArgs={'ACL': 'public-read',
                                  'Metadata': {'source_path': local_paths[i]}})
        print('\n')


def aws_transfer_file(s3, input_bucket, output_bucket, key_src, key_dst):
    """Helper function to transfer files from one bucket/key to another. Used
    in conjunction with a soon-to-be-created transfer jobs script for jobs with multiple stages"""

    copy_source = {'Bucket': output_bucket,
                   'Key': key_src}

    s3.copy(copy_source, input_bucket, key_dst,
            ExtraArgs={'ACL': 'public-read'})


def aws_download_files(upload_log, output_dir):
    """Download files following Figure 8 annotation.

    Args:
        upload_log: pandas file containing information from upload process
        output_dir: directory where files will be saved
    """

    s3 = connect_aws()

    # get files
    files_to_download = upload_log['filename']
    aws_folder = upload_log['aws_folder'][0]
    stage = upload_log['stage'][0]

    # track missing files
    missing = []

    # download all images
    for file in files_to_download:

        # full path to save image
        local_path = os.path.join(output_dir, file)

        # path to file in aws
        aws_path = os.path.join(aws_folder, stage, file)

        try:
            s3.download_file(Bucket='caliban-output', Key=img_path, Filename=save_path)
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print('The file {} does not exist'.format(img))
                missing.append(img)
            else:
                raise e

    return missing
