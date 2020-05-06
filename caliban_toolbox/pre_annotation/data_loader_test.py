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

import errno
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import skimage as sk

import pytest

from deepcell_tracking import tracking
from deepcell_tracking import utils

class TestTracking(object):  # pylint: disable=useless-object-inheritance

    def test_simple(self):
        length = 128
        frames = 3
        x, y = _get_dummy_tracking_data(length, frames=frames)
        num_objects = len(np.unique(y)) - 1
        model = DummyModel()

        _ = tracking.CellTracker(x, y, model=model)

        # test data with bad rank
        with pytest.raises(ValueError):
            tracking.CellTracker(
                np.random.random((32, 32, 1)),
                np.random.randint(num_objects, size=(32, 32, 1)),
                model=model)

        # test mismatched x and y shape
        with pytest.raises(ValueError):
            tracking.CellTracker(
                np.random.random((3, 32, 32, 1)),
                np.random.randint(num_objects, size=(2, 32, 32, 1)),
                model=model)

        # test bad features
        with pytest.raises(ValueError):
            tracking.CellTracker(x, y, model=model, features=None)

        # test bad data_format
        with pytest.raises(ValueError):
            tracking.CellTracker(x, y, model=model, data_format='invalid')