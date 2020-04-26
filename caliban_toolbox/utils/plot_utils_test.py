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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
import xarray as xr

from caliban_toolbox.utils import plot_utils


def test_set_channel_colors():

    # order that colors are plotted in
    color_order = np.array(['red', 'green', 'blue', 'cyan', 'magenta', 'yellow'])

    input_vals = np.random.randint(low=0, high=30, size=(2, 100, 100, 3))

    coords = [['fov1', 'fov2'], range(input_vals.shape[1]), range(input_vals.shape[2]),
              ['chan1', 'chan2', 'chan3']]

    dims = ['fovs', 'rows', 'cols', 'channels']

    input_data = xr.DataArray(input_vals, coords=coords, dims=dims)

    colors = ['magenta', 'green', 'red']

    output_data = plot_utils.set_channel_colors(channel_data=input_data, plot_colors=colors)

    for idx, color in enumerate(colors):
        matched_channel = input_data.channels.values[idx]

        # color is no longer a channel color, indicating it has been assigned to a channel
        assert color not in output_data.channels.values

        color_idx = np.where(np.isin(color_order, color))
        assert output_data.channels.values[color_idx] == matched_channel

    # invalid color
    with pytest.raises(ValueError):
        colors = ['magenta', 'puke', 'red']
        output_data = plot_utils.set_channel_colors(channel_data=input_data, plot_colors=colors)

    # wrong number of colors
    with pytest.raises(ValueError):
        colors = ['magenta', 'blue', 'red', 'yellow']
        output_data = plot_utils.set_channel_colors(channel_data=input_data, plot_colors=colors)

