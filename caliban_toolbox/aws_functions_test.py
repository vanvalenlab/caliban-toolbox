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

from caliban_toolbox import aws_functions
import pathlib


class FakeS3(object):

    def __init__(self, *_, **__):
        pass

    def client(self, *_, **__):
        return self

    def upload_file(self, Filename, Bucket, Key, Callback, ExtraArgs):
        assert os.path.exists(Filename)

    def download_file(self, Bucket, Key, Filename):
        pathlib.Path(Filename).touch()


def test_aws_upload_files(mocker, tmp_path):
    mocker.patch('getpass.getpass', lambda *x: None)
    mocker.patch('boto3.Session', FakeS3)

    local_files = ['npz_file_' + str(num) for num in range(5)]
    aws_paths = ['aws_bucket/folder/npz_file_' + str(num) for num in range(5)]

    for file in local_files:
        pathlib.Path(os.path.join(tmp_path, file)).touch()

    local_paths = [os.path.join(tmp_path, file) for file in local_files]

    aws_functions.aws_upload_files(local_paths=local_paths, aws_paths=aws_paths)


def test_aws_download_files(mocker, tmp_path):
    mocker.patch('getpass.getpass', lambda *x: None)
    mocker.patch('boto3.Session', FakeS3)

    filenames = ['npz_file_' + str(num) for num in range(5)]

    upload_log = {'stage': ['stage_0'],
                  'aws_folder': ['temp_folder'],
                  'filename': filenames}

    aws_functions.aws_download_files(upload_log=upload_log, output_dir=tmp_path)
