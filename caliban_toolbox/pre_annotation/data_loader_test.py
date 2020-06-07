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

import random

import pytest

from caliban_toolbox.pre_annotation import data_loader


def _get_dummy_inputs(object):
    possible_data_type = random.choice([['2d', 'static'],
                                        ['2d', 'dynamic'],
                                        ['3d', 'static'],
                                        ['3d', 'dynamic']])
    possible_imaging_types = random.choice([['fluo'], ['phase'], ['fluo', 'phase'], ['all']])
    possible_specimen_types = random.choice([['HEK293'], ['HeLa'], ['HEK293', 'HeLa'], ['all']])
    possible_compartments = random.choice([[None], ['nuclear'], ['nuclear', 'wholecell'], ['all']])
    possible_markers = ['all']
    possible_exp_ids = ['all']
    possible_sessions = ['all']
    possible_positions = ['all']
    possible_file_type = '.tif'

    loader_inputs = [possible_data_type,
                     possible_imaging_types,
                     possible_specimen_types,
                     possible_compartments,
                     possible_markers,
                     possible_exp_ids,
                     possible_sessions,
                     possible_positions,
                     possible_file_type]

    return loader_inputs


class TestUniversalDataLoader(object):  # pylint: disable=useless-object-inheritance

    def test_simple(self):
        loader_inputs = _get_dummy_inputs(self)

        # test with standard inputs
        _ = data_loader.UniversalDataLoader(data_type=loader_inputs[0],
                                            imaging_types=loader_inputs[1],
                                            specimen_types=loader_inputs[2],
                                            compartments=loader_inputs[3],
                                            markers=loader_inputs[4],
                                            exp_ids=loader_inputs[5],
                                            sessions=loader_inputs[6],
                                            positions=loader_inputs[7],
                                            file_type=loader_inputs[8])
