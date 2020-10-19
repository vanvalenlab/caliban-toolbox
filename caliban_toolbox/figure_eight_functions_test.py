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
import pytest
import json
import requests_mock
import urllib
import pathlib
import zipfile
import tempfile

import numpy as np
import pandas as pd

from pathlib import Path


from caliban_toolbox import figure_eight_functions
from caliban_toolbox.aws_functions_test import FakeS3


class FakeOpen(object):
    def __init__(self, save_path, mode):
        self.save_path = save_path

    def write(self, content):
        save_folder = os.path.dirname(self.save_path)
        pathlib.Path(os.path.join(save_folder, 'example_file.csv')).touch()
        zipfile.ZipFile(self.save_path, mode='w').write(os.path.join(save_folder,
                                                                     'example_file.csv'))


class FakeResponse(object):
    def __init__(self, status_code):
        self.status_code = status_code


def test_copy_job():
    with requests_mock.Mocker() as m:

        # create test data
        test_job_id = 666
        test_appen_key = 'a1b2c3'
        return_id = 123
        return_dict = {'status_code': 200, 'id': return_id}

        # generate same url as function for mocking
        url = 'https://api.appen.com/v1/jobs/{}/copy.json?'.format(str(test_job_id))

        # mock the call
        m.get(url, text=json.dumps(return_dict))
        new_job_id = figure_eight_functions.copy_job(job_id=test_job_id, key=test_appen_key)

        assert new_job_id == return_id


def test_upload_log_file(mocker):
    fake_response = FakeResponse(status_code=200)

    mocker.patch('requests.put', lambda url, data, headers: fake_response)

    # create test data
    data = {'project_url': 'https://caliban.deepcell.org/example_job.npz',
            'stage': 'test'}
    example_log_string = pd.DataFrame(data=data, index=range(1)).to_string()
    test_key = 'a1b2c3'
    test_job_id = 123

    returned_status = figure_eight_functions.upload_log_file(log_file=example_log_string,
                                                             job_id=test_job_id, key=test_key)
    assert returned_status == fake_response.status_code

    # bad status code
    with pytest.raises(ValueError):
        fake_response = FakeResponse(status_code=666)
        mocker.patch('requests.put', lambda url, data, headers: fake_response)

        returned_status = figure_eight_functions.upload_log_file(log_file=example_log_string,
                                                                 job_id=test_job_id,
                                                                 key=test_key)


def test_create_figure_eight_job(mocker, tmp_path):
    mocker.patch('getpass.getpass', lambda *x: 'test_api_key')
    mocker.patch('caliban_toolbox.figure_eight_functions.copy_job', lambda job_id, key: '123')
    mocker.patch('boto3.Session', FakeS3)
    mocker.patch('caliban_toolbox.figure_eight_functions.upload_log_file',
                 lambda log_file, job_id, key: 200)

    # create crop directory
    crop_dir = os.path.join(tmp_path, 'crop_dir')
    os.makedirs(crop_dir)
    np.savez(os.path.join(crop_dir, 'test_crop.npz'))

    status_code = figure_eight_functions.create_figure_eight_job(base_dir=tmp_path,
                                                                 job_id_to_copy=123,
                                                                 aws_folder='aws',
                                                                 stage='stage')
    assert status_code == 200


def test_unzip_report(tmp_path):

    # create example zip file
    pathlib.Path(os.path.join(tmp_path, 'example_file.csv')).touch()
    zip_path = os.path.join(tmp_path, 'job_report.zip')
    zipfile.ZipFile(zip_path, mode='w').write(os.path.join(tmp_path, 'example_file.csv'))

    figure_eight_functions.unzip_report(tmp_path)

    assert os.path.exists(os.path.join(tmp_path, 'job_report.csv'))


def test_download_figure_eight_output(mocker, tmp_path):

    mocker.patch('getpass.getpass', lambda *x: 'example_pass')
    mocker.patch('caliban_toolbox.figure_eight_functions.open', FakeOpen)
    mocker.patch('boto3.Session', FakeS3)

    # create logs directory with upload log
    os.makedirs(os.path.join(tmp_path, 'logs'))
    log_dict = {'job_id': [1234, 1234],
                'filename': ['example_1.npz', 'example_2.npz'],
                'aws_folder': ['example_folder', 'example_folder'],
                'stage': ['stage_0', 'stage_0']
                }

    log_file = pd.DataFrame(log_dict)

    log_file.to_csv(os.path.join(tmp_path, 'logs', 'stage_0_upload_log.csv'))

    missing = figure_eight_functions.download_figure_eight_output(tmp_path)
    assert missing == []

    # catch missing file error, return list of missing files
    mocker.patch('boto3.Session',
                 lambda aws_access_key_id, aws_secret_access_key: FakeS3(raise_error='missing'))
    missing = figure_eight_functions.download_figure_eight_output(tmp_path)
    missing = [os.path.split(file_path)[1] for file_path in missing]
    assert missing == log_dict['filename']
