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
import tempfile
import os
import pytest
import json
import requests_mock
import urllib
import pathlib
import zipfile

import numpy as np
import pandas as pd

from pathlib import Path
from unittest.mock import patch


from caliban_toolbox import figure_eight_functions


def test_get_latest_log_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        upload_logs = ['stage_0_upload_log.csv', 'stage_3_upload_log.csv',
                       'stage_8_upload_log.csv']

        for log in upload_logs:
            Path(os.path.join(temp_dir, log)).touch()

        latest_file = figure_eight_functions.get_latest_log_file(temp_dir)
        assert latest_file == 'stage_8_upload_log.csv'


def test_create_job_urls():
    with tempfile.TemporaryDirectory() as temp_dir:
        npz_files = ['test_0.npz', 'test_1.npz', 'test_2.npz']

        for npz in npz_files:
            Path(os.path.join(temp_dir, npz)).touch()

        aws_folder = 'aws_main_folder/aws_sub_folder'
        stage = 'test_stage'
        pixel_only, label_only, rgb_mode = True, False, True

        output_lists = figure_eight_functions.create_job_urls(crop_dir=temp_dir,
                                                              aws_folder=aws_folder,
                                                              stage=stage,
                                                              pixel_only=pixel_only,
                                                              label_only=label_only,
                                                              rgb_mode=rgb_mode)

        npz_paths, npz_keys, url_paths, npzs_to_upload = output_lists

        assert len(npz_paths) == len(npz_keys) == len(url_paths) == len(npzs_to_upload) == 3

    # TODO: Figure out how we're going to validate inputs
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     # NPZ name with spaces leads to bad url
    #     npz_files = ['bad file name.npz']
    #
    #     for npz in npz_files:
    #         Path(os.path.join(temp_dir, npz)).touch()
    #
    #     aws_folder = 'aws_main_folder/aws_sub_folder'
    #     stage = 'test_stage'
    #     pixel_only, label_only, rgb_mode = True, False, True
    #
    #     with pytest.raises(ValueError):
    #
    #         output_lists = figure_eight_functions.create_job_urls(crop_dir=temp_dir,
    #                                                               aws_folder=aws_folder,
    #                                                               stage=stage,
    #                                                               pixel_only=pixel_only,
    #                                                               label_only=label_only,
    #                                                               rgb_mode=rgb_mode)


# TODO: Is this test useful?
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


def test_upload_log_file():
    with requests_mock.Mocker() as m:

        # create test data
        data = {'project_url': 'https://caliban.deepcell.org/example_job.npz',
                'stage': 'test'}
        example_log_string = pd.DataFrame(data=data, index=range(1)).to_string()
        test_key = 'a1b2c3'
        test_job_id = 123

        # generate same url as function for mocking
        url = "https://api.appen.com/v1/jobs/{}/upload.json?{}"
        url_dict = {'key': test_key, 'force': True}
        url_encoded_dict = urllib.parse.urlencode(url_dict)
        url = url.format(test_job_id, url_encoded_dict)

        # mock the call
        response_dict = {'status_code': 200}
        m.put(url, text=json.dumps(response_dict))
        figure_eight_functions.upload_log_file(log_file=example_log_string, job_id=test_job_id,
                                               key=test_key)

# TODO: This mocks almost every call within the function, so we don't get proper integration
# testing of the different parts working together. However, better than not testing at all


@patch('caliban_toolbox.figure_eight_functions.upload_log_file')
@patch('caliban_toolbox.figure_eight_functions.aws_upload_files')
@patch('caliban_toolbox.figure_eight_functions.copy_job')
@patch("getpass.getpass")
def test_create_figure_eight_job(getpass, copy_job, aws_upload_files, upload_log_file):
    getpass.return_value = 'test_api_key'
    copy_job.return_value = 123
    aws_upload_files.return_value = '200'
    upload_log_file.return_value = 567

    with tempfile.TemporaryDirectory() as temp_dir:

        # create crop directory
        crop_dir = os.path.join(temp_dir, 'crop_dir')
        os.makedirs(crop_dir)
        np.savez(os.path.join(crop_dir, 'test_crop.npz'))

        figure_eight_functions.create_figure_eight_job(base_dir=temp_dir, job_id_to_copy=123,
                                                       aws_folder='aws', stage='stage')


def test_unzip_report():
    with tempfile.TemporaryDirectory() as temp_dir:
        # create example zip file
        pathlib.Path(os.path.join(temp_dir, 'example_file.csv')).touch()
        zip_path = os.path.join(temp_dir, 'job_report.zip')
        zipfile.ZipFile(zip_path, mode='w').write(os.path.join(temp_dir, 'example_file.csv'))

        figure_eight_functions.unzip_report(temp_dir)

        assert os.path.exists(os.path.join(temp_dir, 'job_report.csv'))


@patch('caliban_toolbox.figure_eight_functions.aws_download_files')
@patch('caliban_toolbox.figure_eight_functions.download_report')
def test_download_figure_eight_output(download_report, aws_download_files):

    # we don't care about this return value, this is just to override existing function
    download_report.return_value = 200
    aws_download_files.return_value = 200

    with tempfile.TemporaryDirectory() as temp_dir:

        # create logs directory with zipped report
        os.makedirs(os.path.join(temp_dir, 'logs'))
        pathlib.Path(os.path.join(temp_dir, 'logs', 'example_file.csv')).touch()
        zip_path = os.path.join(temp_dir, 'logs', 'job_report.zip')
        zipfile.ZipFile(zip_path, mode='w').write(os.path.join(temp_dir, 'logs',
                                                               'example_file.csv'))
        # create log file
        log_file = pd.DataFrame({'job_id': [1234]})
        log_file.to_csv(os.path.join(temp_dir, 'logs', 'stage_0_upload_log.csv'))

        figure_eight_functions.download_figure_eight_output(temp_dir)
