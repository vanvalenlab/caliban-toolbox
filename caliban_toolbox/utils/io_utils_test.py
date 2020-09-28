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
import json
import pytest
import copy
import tempfile

import numpy as np

from caliban_toolbox import reshape_data
from caliban_toolbox.utils import io_utils

from caliban_toolbox.utils.crop_utils_test import _blank_data_xr


def test_save_npzs_for_caliban():
    fov_len, stack_len, num_crops, num_slices, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 4

    X_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=num_crops,
                            slice_num=num_slices,
                            row_len=row_len, col_len=col_len, chan_len=chan_len)

    y_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=num_crops,
                            slice_num=num_slices,
                            row_len=row_len, col_len=col_len, chan_len=1,
                            last_dim_name='compartments')

    sliced_X, sliced_y, log_data = reshape_data.create_slice_data(X_data=X_data, y_data=y_data,
                                                                  slice_stack_len=slice_stack_len)

    with tempfile.TemporaryDirectory() as temp_dir:
        io_utils.save_npzs_for_caliban(X_data=sliced_X, y_data=sliced_y, original_data=X_data,
                                       log_data=copy.copy(log_data), save_dir=temp_dir,
                                       blank_labels="include",
                                       save_format="npz", verbose=False)

        # check that correct size was saved
        test_npz_labels = np.load(os.path.join(temp_dir, "fov_fov1_crop_0_slice_0.npz"))

        assert test_npz_labels["y"].shape == (slice_stack_len, row_len, col_len, 1)

        assert test_npz_labels["y"].shape[:-1] == test_npz_labels["X"].shape[:-1]

        # check that json saved successfully
        with open(os.path.join(temp_dir, "log_data.json")) as json_file:
            saved_log_data = json.load(json_file)

        assert saved_log_data["original_shape"] == list(X_data.shape)

    with tempfile.TemporaryDirectory() as temp_dir:
        # check that combined crop and slice saving works
        crop_size = (10, 10)
        overlap_frac = 0.2
        X_cropped, y_cropped, log_data_crop = \
            reshape_data.crop_multichannel_data(X_data=sliced_X,
                                                y_data=sliced_y,
                                                crop_size=crop_size,
                                                overlap_frac=overlap_frac,
                                                test_parameters=False)

        io_utils.save_npzs_for_caliban(X_data=X_cropped, y_data=y_cropped,
                                       original_data=X_data,
                                       log_data={**log_data, **log_data_crop},
                                       save_dir=temp_dir,
                                       blank_labels="include", save_format="npz",
                                       verbose=False)
        expected_crop_num = X_cropped.shape[2] * X_cropped.shape[3]
        files = os.listdir(temp_dir)
        files = [file for file in files if "npz" in file]

        assert len(files) == expected_crop_num

    # check that arguments specifying what to do with blank crops are working

    # set specified crops to not be blank
    sliced_y[0, 0, 0, [1, 4, 7], 0, 0, 0] = 27
    expected_crop_num = sliced_X.shape[2] * sliced_X.shape[3]

    # test that function correctly includes blank crops when saving
    with tempfile.TemporaryDirectory() as temp_dir:
        io_utils.save_npzs_for_caliban(X_data=sliced_X, y_data=sliced_y,
                                       original_data=X_data,
                                       log_data=copy.copy(log_data), save_dir=temp_dir,
                                       blank_labels="include",
                                       save_format="npz", verbose=False)

        # check that there is the expected number of files saved to directory
        files = os.listdir(temp_dir)
        files = [file for file in files if "npz" in file]

        assert len(files) == expected_crop_num

    # test that function correctly skips blank crops when saving
    with tempfile.TemporaryDirectory() as temp_dir:
        io_utils.save_npzs_for_caliban(X_data=sliced_X, y_data=sliced_y,
                                       original_data=X_data,
                                       log_data=copy.copy(log_data), save_dir=temp_dir,
                                       save_format="npz",
                                       blank_labels="skip", verbose=False)

        #  check that expected number of files in directory
        files = os.listdir(temp_dir)
        files = [file for file in files if "npz" in file]
        assert len(files) == 3

    # test that function correctly saves blank crops to separate folder
    with tempfile.TemporaryDirectory() as temp_dir:
        io_utils.save_npzs_for_caliban(X_data=sliced_X, y_data=sliced_y,
                                       original_data=X_data,
                                       log_data=copy.copy(log_data), save_dir=temp_dir,
                                       save_format="npz",
                                       blank_labels="separate", verbose=False)

        # check that expected number of files in each directory
        files = os.listdir(temp_dir)
        files = [file for file in files if "npz" in file]
        assert len(files) == 3

        files = os.listdir(os.path.join(temp_dir, "separate"))
        files = [file for file in files if "npz" in file]
        assert len(files) == expected_crop_num - 3


# postprocessing


def test_get_npz_file_path():
    # create list of npz_ids
    dir_list = ["fov_fov1_crop_2_slice_4.npz", "fov_fov1_crop_2_slice_5_save_version_0.npz",
                "fov_fov1_crop_2_slice_6_save_version_0.npz",
                "fov_fov1_crop_2_slice_6_save_version_1.npz",
                "fov_fov1_crop_2_slice_7_save_version_0.npz",
                "fov_fov1_crop_2_slice_7_save_version_0_save_version_2.npz"]

    fov, crop = "fov1", 2

    # test unmodified npz
    slice = 4
    output_string = io_utils.get_saved_file_path(dir_list, fov, crop, slice)

    assert output_string == dir_list[0]

    # test single modified npz
    slice = 5
    output_string = io_utils.get_saved_file_path(dir_list, fov, crop, slice)
    assert output_string == dir_list[1]

    # test that error is raised when multiple save versions present
    slice = 6
    with pytest.raises(ValueError):
        output_string = io_utils.get_saved_file_path(dir_list, fov, crop, slice)

    # test that error is raised when multiple save versions present due to resaves
    slice = 7

    with pytest.raises(ValueError):
        output_string = io_utils.get_saved_file_path(dir_list, fov, crop, slice)


def test_load_npzs():
    with tempfile.TemporaryDirectory() as temp_dir:
        # first generate image stack that will be sliced up
        fov_len, stack_len, crop_num, slice_num = 1, 40, 1, 1
        row_len, col_len, chan_len = 50, 50, 3
        slice_stack_len = 4

        X_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

        y_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=1,
                                last_dim_name='compartments')

        # slice the data
        X_slice, y_slice, log_data = reshape_data.create_slice_data(X_data, y_data,
                                                                    slice_stack_len)

        # crop the data
        crop_size = (10, 10)
        overlap_frac = 0.2
        X_cropped, y_cropped, log_data_crop = \
            reshape_data.crop_multichannel_data(
                X_data=X_slice,
                y_data=y_slice,
                crop_size=crop_size,
                overlap_frac=overlap_frac,
                test_parameters=False)

        # tag the upper left hand corner of the label in each slice
        slice_tags = np.arange(y_cropped.shape[3])
        crop_tags = np.arange(y_cropped.shape[2])
        y_cropped[0, 0, :, 0, 0, 0, 0] = crop_tags
        y_cropped[0, 0, 0, :, 0, 0, 0] = slice_tags

        combined_log_data = {**log_data, **log_data_crop}

        # save the tagged data
        io_utils.save_npzs_for_caliban(X_data=X_cropped, y_data=y_cropped, original_data=X_data,
                                       log_data=combined_log_data, save_dir=temp_dir,
                                       blank_labels="include", save_format="npz",
                                       verbose=False)

        with open(os.path.join(temp_dir, "log_data.json")) as json_file:
            saved_log_data = json.load(json_file)

        loaded_slices = io_utils.load_npzs(temp_dir, saved_log_data, verbose=False)

        # dims other than channels are the same
        assert (np.all(loaded_slices.shape[:-1] == X_cropped.shape[:-1]))

        assert np.all(np.equal(loaded_slices[0, 0, :, 0, 0, 0, 0], crop_tags))
        assert np.all(np.equal(loaded_slices[0, 0, 0, :, 0, 0, 0], slice_tags))

    # test slices with unequal last length
    with tempfile.TemporaryDirectory() as temp_dir:
        # first generate image stack that will be sliced up
        fov_len, stack_len, crop_num, slice_num = 1, 40, 1, 1
        row_len, col_len, chan_len = 50, 50, 3
        slice_stack_len = 7

        X_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

        y_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=1,
                                last_dim_name='compartments')

        # slice the data
        X_slice, y_slice, log_data = reshape_data.create_slice_data(X_data,
                                                                    y_data,
                                                                    slice_stack_len)

        # crop the data
        crop_size = (10, 10)
        overlap_frac = 0.2
        X_cropped, y_cropped, log_data_crop = \
            reshape_data.crop_multichannel_data(
                X_data=X_slice,
                y_data=y_slice,
                crop_size=crop_size,
                overlap_frac=overlap_frac,
                test_parameters=False)

        # tag the upper left hand corner of the annotations in each slice
        slice_tags = np.arange(y_cropped.shape[3])
        crop_tags = np.arange(X_cropped.shape[2])
        y_cropped[0, 0, :, 0, 0, 0, 0] = crop_tags
        y_cropped[0, 0, 0, :, 0, 0, 0] = slice_tags

        combined_log_data = {**log_data, **log_data_crop}

        # save the tagged data
        io_utils.save_npzs_for_caliban(X_data=X_cropped, y_data=y_cropped, original_data=X_data,
                                       log_data=combined_log_data, save_dir=temp_dir,
                                       blank_labels="include", save_format="npz",
                                       verbose=False)

        loaded_slices = io_utils.load_npzs(temp_dir, combined_log_data)

        # dims other than channels are the same
        assert (np.all(loaded_slices.shape[:-1] == X_cropped.shape[:-1]))

        assert np.all(np.equal(loaded_slices[0, 0, :, 0, 0, 0, 0], crop_tags))
        assert np.all(np.equal(loaded_slices[0, 0, 0, :, 0, 0, 0], slice_tags))
