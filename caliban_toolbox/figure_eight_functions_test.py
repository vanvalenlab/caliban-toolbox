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

from pathlib import Path


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

    with tempfile.TemporaryDirectory() as temp_dir:
        # NPZ name with spaces leads to bad url
        npz_files = ['bad file name.npz']

        for npz in npz_files:
            Path(os.path.join(temp_dir, npz)).touch()

        aws_folder = 'aws_main_folder/aws_sub_folder'
        stage = 'test_stage'
        pixel_only, label_only, rgb_mode = True, False, True

        with pytest.raises(ValueError):

            output_lists = figure_eight_functions.create_job_urls(crop_dir=temp_dir,
                                                                  aws_folder=aws_folder,
                                                                  stage=stage,
                                                                  pixel_only=pixel_only,
                                                                  label_only=label_only,
                                                                  rgb_mode=rgb_mode)
