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

from unittest.mock import patch
from caliban_toolbox import aws_functions
import tempfile
import pathlib


# TODO: What is a better way to mock the s3 = connect_aws() call within this function
@patch('caliban_toolbox.aws_functions.connect_aws')
def test_aws_upload_files(connect_aws):

    class Fake_S3(object):
        def upload_file(self, x1, x2, x3, Callback, ExtraArgs):
            pass

    f_s3 = Fake_S3()

    connect_aws.return_value = f_s3
    local_files = ['npz_file_' + str(num) for num in range(5)]
    aws_paths = ['aws_bucket/folder/npz_file_' + str(num) for num in range(5)]

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in local_files:
            pathlib.Path(os.path.join(temp_dir, file)).touch()

        local_paths = [os.path.join(temp_dir, file) for file in local_files]

        aws_functions.aws_upload_files(local_paths=local_paths, aws_paths=aws_paths)


@patch('caliban_toolbox.aws_functions.connect_aws')
def test_aws_download_files(connect_aws):

    class Fake_S3(object):
        def download_file(self, Bucket, Key, Filename):
            pass

    f_s3 = Fake_S3()

    connect_aws.return_value = f_s3

    aws_paths = ['aws_bucket/folder/npz_file_' + str(num) for num in range(5)]

    upload_log = {'stage': ['stage_0'],
                  'aws_folder': ['temp_folder'],
                  'filename': aws_paths}

    output_dir = 'example/output/dir'

    aws_functions.aws_download_files(upload_log=upload_log, output_dir=output_dir)
