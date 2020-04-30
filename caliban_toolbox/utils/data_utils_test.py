# Copyright 2016-2020 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the 'License');
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
# distributed under the License is distributed on an 'AS IS' BASIS,
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
from caliban_toolbox.utils import data_utils


def test_pad_xr_dims():
    test_input = np.zeros((2, 10, 10, 3))
    test_coords = [['Point1', 'Point2'], range(test_input.shape[1]), range(test_input.shape[2]),
                   ['chan0', 'chan1', 'chan2']]
    test_dims = ['fovs', 'rows', 'cols', 'channels']

    test_xr = xr.DataArray(test_input, coords=test_coords, dims=test_dims)

    padded_dims = ['fovs', 'rows', 'new_dim1', 'cols', 'new_dim2', 'channels']

    padded_xr = data_utils.pad_xr_dims(test_xr, padded_dims)

    assert list(padded_xr.dims) == padded_dims
    assert len(padded_xr['new_dim1']) == len(padded_xr['new_dim2']) == 1

    # check that error raised when wrong dimensions
    padded_wrong_order_dims = ['rows', 'fovs', 'new_dim1', 'cols', 'new_dim2', 'channels']

    with pytest.raises(ValueError):
        data_utils.pad_xr_dims(test_xr, padded_wrong_order_dims)

    # check that error raised when duplicated dimensions
    padded_duplicated_dims = ['fovs', 'rows', 'new_dim1', 'cols', 'new_dim1', 'channels']

    with pytest.raises(ValueError):
        data_utils.pad_xr_dims(test_xr, padded_duplicated_dims)


def test_create_blank_channel():

    semi_blank = data_utils.create_blank_channel(img_size=(1024, 1024), dtype="int16",
                                                 full_blank=False)

    assert semi_blank.shape == (1024, 1024)
    assert np.sum(semi_blank > 0) == 10 * 10

    full_blank = data_utils.create_blank_channel(img_size=(1024, 1024), dtype="int16",
                                                 full_blank=True)
    assert np.sum(full_blank) == 0


def test_reorder_channels():

    # test switching without blank channels
    test_input = np.random.randint(5, size=(2, 128, 128, 3))

    # channel 0 is 3x bigger, channel 2 is 3x smaller
    test_input[:, :, :, 0] *= 3
    test_input[:, :, :, 2] //= 3

    coords = [['fov1', 'fov2'], range(test_input.shape[1]),
              range(test_input.shape[2]), ['chan0', 'chan1', 'chan2']]
    dims = ['fovs', 'rows', 'cols', 'channels']
    input_data = xr.DataArray(test_input, coords=coords, dims=dims)

    new_channel_order = ['chan2', 'chan1', 'chan0']

    reordered_data = data_utils.reorder_channels(new_channel_order=new_channel_order,
                                                 input_data=input_data)

    # confirm that labels are in correct order, and that values were switched as well
    assert np.array_equal(new_channel_order, reordered_data.channels)

    for chan in input_data.channels.values:
        assert np.array_equal(reordered_data.loc[:, :, :, chan], input_data.loc[:, :, :, chan])

    # test switching with blank channels
    new_channel_order = ['chan0', 'chan1', 'chan666', 'chan2']
    reordered_data = data_utils.reorder_channels(new_channel_order=new_channel_order,
                                                 input_data=input_data)

    # make sure order was switched, and that blank channel is empty
    assert np.array_equal(new_channel_order, reordered_data.channels)

    # make sure new channel is empty and existing channels have same value
    for chan in reordered_data.channels.values:
        if chan in input_data.channels:
            assert np.array_equal(reordered_data.loc[:, :, :, chan], input_data.loc[:, :, :, chan])
        else:
            assert np.sum(reordered_data.loc[:, :, :, chan].values > 0) == 0

    # test switching with blank channels and existing channels in new order
    new_channel_order = ['chan2', 'chan11', 'chan1', 'chan12', 'chan0']
    reordered_data = data_utils.reorder_channels(new_channel_order=new_channel_order,
                                                 input_data=input_data)

    assert np.array_equal(new_channel_order, reordered_data.channels)

    # make sure new channel is empty and existing channels have same value
    for chan in reordered_data.channels.values:
        if chan in input_data.channels:
            assert np.array_equal(reordered_data.loc[:, :, :, chan], input_data.loc[:, :, :, chan])
        else:
            assert np.sum(reordered_data.loc[:, :, :, chan].values > 0) == 0

    # New channels have duplicates
    with pytest.raises(ValueError):
        new_channel_order = ['chan0', 'chan1', 'chan2', 'chan2']
        reordered_data = data_utils.reorder_channels(new_channel_order=new_channel_order,
                                                     input_data=input_data)


def test_make_blank_labels():
    assert True
