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
"""Tests for data_loader.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import skimage as sk

import pytest

from caliban_toolbox.pre_annotation import data_loader

def _get_dummy_inputs(object):
    possible_data_type = [['2d', 'static'], ['2d', 'dynamic'], ['3d', 'static'], ['3d', 'dynamic']]
    possible_imaging_types = [[]]
    possible_specimen_types
    possible_compartments=None
    possible_markers=['all']
    possible_exp_ids=['all']
    possible_sessions=['all']
    possible_positions=['all']
    possible_file_type='.tif'


class TestUniversalDataLoader(object):  # pylint: disable=useless-object-inheritance

    def test_simple(self):
        loader_inputs = _get_dummy_inputs(self)
        _ = data_loader.UniversalDataLoader(loader_inputs)

        # test data with bad rank
        with pytest.raises(ValueError):
            data_loader.UniversalDataLoader(
                np.random.random((32, 32, 1)),
                np.random.randint(num_objects, size=(32, 32, 1)),
                model=model)

        # test mismatched x and y shape
        with pytest.raises(ValueError):
            data_loader.UniversalDataLoader(
                np.random.random((3, 32, 32, 1)),
                np.random.randint(num_objects, size=(2, 32, 32, 1)),
                model=model)

        # test bad features
        with pytest.raises(ValueError):
            data_loader.UniversalDataLoader(x, y, model=model, features=None)

        # test bad data_format
        with pytest.raises(ValueError):
            data_loader.UniversalDataLoader(x, y, model=model, data_format='invalid')