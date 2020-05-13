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
import tempfile

import numpy as np

from caliban_toolbox import pipeline_utils
import importlib

importlib.reload(pipeline_utils)


def _make_raw_metadata():
    metadata_file = {'PROJECT_ID': np.random.randint(1, 100),
                     'EXPERIMENT_ID': np.random.randint(1, 100)}

    return metadata_file


def _make_fov_ids(num_fovs):
    all_fovs = np.random.randint(low=1, high=num_fovs*10, size=num_fovs)
    fovs = ['fov_{}'.format(i) for i in all_fovs]

    return fovs


def test_get_job_folder_name():
    # folder already exists
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_name = 'caliban_job_0'
        os.makedirs(os.path.join(temp_dir, folder_name))

        _, folder_name = pipeline_utils.get_job_folder_name(temp_dir)
        assert folder_name == 'caliban_job_1'

    # first folder
    with tempfile.TemporaryDirectory() as temp_dir:
        _, folder_name = pipeline_utils.get_job_folder_name(temp_dir)
        assert folder_name == 'caliban_job_0'

