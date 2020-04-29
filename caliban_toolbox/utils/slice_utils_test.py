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
import numpy as np

from caliban_toolbox import reshape_data
from caliban_toolbox.utils import slice_utils

from caliban_toolbox.utils.crop_utils_test import _blank_data_xr


def test_compute_slice_indices():
    # test when slice divides evenly into stack len
    stack_len = 40
    slice_len = 4
    slice_overlap = 0
    slice_start_indices, slice_end_indices = slice_utils.compute_slice_indices(stack_len,
                                                                               slice_len,
                                                                               slice_overlap)
    assert np.all(np.equal(slice_start_indices, np.arange(0, stack_len, slice_len)))

    # test when slice_num does not divide evenly into stack_len
    stack_len = 42
    slice_len = 5
    slice_start_indices, slice_end_indices = slice_utils.compute_slice_indices(stack_len,
                                                                               slice_len,
                                                                               slice_overlap)

    expected_start_indices = np.arange(0, stack_len, slice_len)
    assert np.all(np.equal(slice_start_indices, expected_start_indices))

    # test overlapping slices
    stack_len = 40
    slice_len = 4
    slice_overlap = 1
    slice_start_indices, slice_end_indices = slice_utils.compute_slice_indices(stack_len,
                                                                               slice_len,
                                                                               slice_overlap)

    assert len(slice_start_indices) == int(np.floor(stack_len / (slice_len - slice_overlap)))
    assert slice_end_indices[-1] == stack_len
    assert slice_end_indices[0] - slice_start_indices[0] == slice_len


def test_slice_helper():
    # test output shape with even division of slice
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 4

    slice_start_indices, slice_end_indices = slice_utils.compute_slice_indices(stack_len,
                                                                               slice_stack_len, 0)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num, row_len=row_len, col_len=col_len,
                                chan_len=chan_len)

    slice_output = slice_utils.slice_helper(input_data, slice_start_indices, slice_end_indices)

    assert slice_output.shape == (fov_len, slice_stack_len, crop_num,
                                  int(np.ceil(stack_len / slice_stack_len)),
                                  row_len, col_len, chan_len)

    # test output shape with uneven division of slice
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 6

    slice_start_indices, slice_end_indices = slice_utils.compute_slice_indices(stack_len,
                                                                               slice_stack_len, 0)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num, row_len=row_len, col_len=col_len,
                                chan_len=chan_len)

    slice_output = slice_utils.slice_helper(input_data, slice_start_indices, slice_end_indices)

    assert slice_output.shape == (fov_len, slice_stack_len, crop_num,
                                  (np.ceil(stack_len / slice_stack_len)),
                                  row_len, col_len, chan_len)

    # test output shape with slice overlaps
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 6
    slice_overlap = 1
    slice_start_indices, slice_end_indices = slice_utils.compute_slice_indices(stack_len,
                                                                               slice_stack_len,
                                                                               slice_overlap)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num, row_len=row_len, col_len=col_len,
                                chan_len=chan_len)

    slice_output = slice_utils.slice_helper(input_data, slice_start_indices, slice_end_indices)

    assert slice_output.shape == (fov_len, slice_stack_len, crop_num,
                                  (np.ceil(stack_len / (slice_stack_len - slice_overlap))),
                                  row_len, col_len, chan_len)

    # test output values
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 4
    slice_start_indices, slice_end_indices = slice_utils.compute_slice_indices(stack_len,
                                                                               slice_stack_len, 0)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num, row_len=row_len, col_len=col_len,
                                chan_len=chan_len)

    # tag upper left hand corner of each image
    tags = np.arange(stack_len)
    input_data[0, :, 0, 0, 0, 0, 0] = tags

    slice_output = slice_utils.slice_helper(input_data, slice_start_indices, slice_end_indices)

    # loop through each slice, make sure values increment as expected
    for i in range(slice_output.shape[1]):
        assert np.all(np.equal(slice_output[0, :, 0, i, 0, 0, 0], tags[i * 4:(i + 1) * 4]))


def test_stitch_slices():
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 4

    X_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                            slice_num=slice_num,
                            row_len=row_len, col_len=col_len, chan_len=chan_len)

    y_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                            slice_num=slice_num,
                            row_len=row_len, col_len=col_len, chan_len=1)

    # generate ordered data
    linear_seq = np.arange(stack_len * row_len * col_len)
    test_vals = linear_seq.reshape((stack_len, row_len, col_len))
    y_data[0, :, 0, 0, :, :, 0] = test_vals

    X_slice, y_slice, log_data = reshape_data.create_slice_data(X_data=X_data, y_data=y_data,
                                                                slice_stack_len=slice_stack_len)

    log_data["original_shape"] = X_data.shape
    log_data["fov_names"] = X_data.fovs.values
    stitched_slices = slice_utils.stitch_slices(y_slice, {**log_data})

    # dims are the same
    assert np.all(stitched_slices.shape == y_data.shape)

    assert np.all(np.equal(stitched_slices[0, :, 0, 0, :, :, 0], test_vals))

    # test case without even division of crops into imsize

    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 7

    X_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                            slice_num=slice_num,
                            row_len=row_len, col_len=col_len, chan_len=chan_len)

    y_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                            slice_num=slice_num,
                            row_len=row_len, col_len=col_len, chan_len=1)

    # generate ordered data
    linear_seq = np.arange(stack_len * row_len * col_len)
    test_vals = linear_seq.reshape((stack_len, row_len, col_len))
    y_data[0, :, 0, 0, :, :, 0] = test_vals

    X_slice, y_slice, log_data = reshape_data.create_slice_data(X_data=X_data, y_data=y_data,
                                                                slice_stack_len=slice_stack_len)

    # get parameters
    log_data["original_shape"] = y_data.shape
    log_data["fov_names"] = y_data.fovs.values
    stitched_slices = slice_utils.stitch_slices(y_slice, log_data)

    assert np.all(stitched_slices.shape == y_data.shape)

    assert np.all(np.equal(stitched_slices[0, :, 0, 0, :, :, 0], test_vals))
