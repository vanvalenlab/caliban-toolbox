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

import numpy as np
import pandas as pd

from caliban_toolbox import anolytics
from caliban_toolbox.aws_functions_test import FakeS3


def test_create_anolytics_job(mocker, tmp_path):
    mocker.patch('boto3.Session', FakeS3)

    # create crop directory
    crop_dir = os.path.join(tmp_path, 'crop_dir')
    os.makedirs(crop_dir)
    np.savez(os.path.join(crop_dir, 'test_crop.npz'))

    anolytics.create_anolytics_job(base_dir=tmp_path,
                                   aws_folder='aws',
                                   stage='stage')

    test_log_name = os.path.join(tmp_path, 'logs', 'stage_0_{}_upload_log.csv'.format('stage'))
    assert os.path.exists(test_log_name)


def test_download_anolytics_output(mocker, tmp_path):
    mocker.patch('boto3.Session', FakeS3)

    # create logs directory with upload log
    os.makedirs(os.path.join(tmp_path, 'logs'))
    log_dict = {'filename': ['example_1.npz', 'example_2.npz'],
                'aws_folder': ['example_folder', 'example_folder'],
                'stage': ['stage_0', 'stage_0']
                }

    log_file = pd.DataFrame(log_dict)

    log_file.to_csv(os.path.join(tmp_path, 'logs', 'stage_0_upload_log.csv'))

    missing = anolytics.download_anolytics_output(tmp_path)
    assert missing == []

    # catch missing file error, return list of missing files
    mocker.patch('boto3.Session',
                 lambda aws_access_key_id, aws_secret_access_key: FakeS3(raise_error='missing'))
    missing = anolytics.download_anolytics_output(tmp_path)
    missing = [os.path.split(file_path)[1] for file_path in missing]
    assert missing == log_dict['filename']

