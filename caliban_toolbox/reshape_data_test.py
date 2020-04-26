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
import shutil
import json
import pytest
import copy
import tempfile
import skimage.measure

import numpy as np
from caliban_toolbox import reshape_data
import xarray as xr

import importlib

importlib.reload(reshape_data)


def _blank_data_xr(fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len):
    """Test function to generate a blank xarray with the supplied dimensions

    Inputs
        fov_num: number of distinct FOVs
        stack_num: number of distinct z stacks
        crop_num: number of x/y crops
        slice_num: number of z/t slices
        row_num: number of rows
        col_num: number of cols
        chan_num: number of channels

    Outputs
        test_xr: xarray of [fov_num, row_num, col_num, chan_num]"""

    test_img = np.zeros((fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len))

    fovs = ["fov" + str(x) for x in range(1, fov_len + 1)]
    channels = ["channel" + str(x) for x in range(1, chan_len + 1)]

    test_stack_xr = xr.DataArray(data=test_img,
                                 coords=[fovs, range(stack_len), range(crop_num), range(slice_num),
                                         range(row_len), range(col_len), channels],
                                 dims=["fovs", "stacks", "crops", "slices",
                                       "rows", "cols", "channels"])

    return test_stack_xr


def test_compute_crop_indices():
    # test corner case of only one crop
    img_len, crop_size, overlap_frac = 100, 100, 0.2
    starts, ends, padding = reshape_data.compute_crop_indices(img_len=img_len, crop_size=crop_size,
                                                              overlap_frac=overlap_frac)
    assert (len(starts) == 1)
    assert (len(ends) == 1)

    # test crop size that doesn't divide evenly into image size
    img_len, crop_size, overlap_frac = 105, 20, 0.2
    starts, ends, padding = reshape_data.compute_crop_indices(img_len=img_len, crop_size=crop_size,
                                                              overlap_frac=overlap_frac)
    crop_num = np.ceil(img_len / (crop_size - (crop_size * overlap_frac)))
    assert (len(starts) == crop_num)
    assert (len(ends) == crop_num)

    crop_end = crop_num * (crop_size - (crop_size * overlap_frac)) + crop_size * overlap_frac
    assert (ends[-1] == crop_end)

    # test overlap of 0 between crops
    img_len, crop_size, overlap_frac = 200, 20, 0
    starts, ends, padding = reshape_data.compute_crop_indices(img_len=img_len, crop_size=crop_size,
                                                              overlap_frac=overlap_frac)
    assert (np.all(starts == range(0, 200, 20)))
    assert (np.all(ends == range(20, 201, 20)))
    assert (padding == 0)


def test_crop_helper():
    # img params
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 2, 1, 1, 1, 200, 200, 1
    crop_size, overlap_frac = 200, 0.2

    # test only one crop
    test_xr = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                             slice_num=slice_num, row_len=row_len, col_len=col_len,
                             chan_len=chan_len)

    starts, ends, padding = reshape_data.compute_crop_indices(img_len=row_len, crop_size=crop_size,
                                                              overlap_frac=overlap_frac)
    cropped, padded = reshape_data.crop_helper(input_data=test_xr, row_starts=starts,
                                               row_ends=ends, col_starts=starts, col_ends=ends,
                                               padding=(padding, padding))

    assert (cropped.shape == (fov_len, stack_len, 1, slice_num, row_len, col_len, chan_len))

    # test crops of different row/col dimensions
    row_crop, col_crop = 50, 40
    row_starts, row_ends, row_padding = \
        reshape_data.compute_crop_indices(img_len=row_len, crop_size=row_crop,
                                          overlap_frac=overlap_frac)

    col_starts, col_ends, col_padding = \
        reshape_data.compute_crop_indices(img_len=col_len, crop_size=col_crop,
                                          overlap_frac=overlap_frac)

    cropped, padded = reshape_data.crop_helper(input_data=test_xr, row_starts=row_starts,
                                               row_ends=row_ends, col_starts=col_starts,
                                               col_ends=col_ends,
                                               padding=(row_padding, col_padding))

    assert (cropped.shape == (fov_len, stack_len, 30, slice_num, row_crop, col_crop, chan_len))

    # test that correct region of image is being cropped
    row_crop, col_crop = 40, 40

    # assign each pixel in the image a unique value
    linear_sequence = np.arange(0, fov_len * 1 * 1 * row_len * col_len * chan_len)
    linear_sequence_reshaped = np.reshape(linear_sequence, (fov_len, 1, 1, 1, row_len,
                                                            col_len, chan_len))
    test_xr[:, :, :, :, :, :, :] = linear_sequence_reshaped

    # crop the image
    row_starts, row_ends, row_padding = \
        reshape_data.compute_crop_indices(img_len=row_len, crop_size=row_crop,
                                          overlap_frac=overlap_frac)

    col_starts, col_ends, col_padding = \
        reshape_data.compute_crop_indices(img_len=col_len, crop_size=col_crop,
                                          overlap_frac=overlap_frac)

    cropped, padded = reshape_data.crop_helper(input_data=test_xr, row_starts=row_starts,
                                               row_ends=row_ends, col_starts=col_starts,
                                               col_ends=col_ends,
                                               padding=(row_padding, col_padding))

    # check that the values of each crop match the value in uncropped image
    for img in range(test_xr.shape[0]):
        crop_counter = 0
        for row in range(len(row_starts)):
            for col in range(len(col_starts)):
                crop = cropped[img, 0, crop_counter, 0, :, :, 0].values

                original_image_crop = test_xr[img, 0, 0, 0, row_starts[row]:row_ends[row],
                                              col_starts[col]:col_ends[col], 0].values
                assert (np.all(crop == original_image_crop))

                crop_counter += 1


def test_crop_multichannel_data():
    # img params
    fov_len, stack_len, crop_num, slice_num, row_len = 2, 1, 1, 1, 200
    col_len, channel_len = 200, 1
    crop_size = (50, 50)
    overlap_frac = 0.2

    # test only one crop
    test_X_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                 slice_num=slice_num, row_len=row_len, col_len=col_len,
                                 chan_len=channel_len)

    test_y_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                 slice_num=slice_num, row_len=row_len, col_len=col_len,
                                 chan_len=channel_len)

    X_data_cropped, y_data_cropped, log_data = \
        reshape_data.crop_multichannel_data(X_data=test_X_data,
                                            y_data=test_y_data,
                                            crop_size=crop_size,
                                            overlap_frac=overlap_frac,
                                            test_parameters=False)

    expected_crop_num = len(reshape_data.compute_crop_indices(row_len, crop_size[0],
                                                              overlap_frac)[0]) ** 2
    assert (X_data_cropped.shape == (fov_len, stack_len, expected_crop_num, slice_num,
                                      crop_size[0], crop_size[1], channel_len))

    assert log_data["num_crops"] == expected_crop_num


def test_compute_slice_indices():
    # test when slice divides evenly into stack len
    stack_len = 40
    slice_len = 4
    slice_overlap = 0
    slice_start_indices, slice_end_indices = reshape_data.compute_slice_indices(stack_len,
                                                                                slice_len,
                                                                                slice_overlap)
    assert np.all(np.equal(slice_start_indices, np.arange(0, stack_len, slice_len)))

    # test when slice_num does not divide evenly into stack_len
    stack_len = 42
    slice_len = 5
    slice_start_indices, slice_end_indices = reshape_data.compute_slice_indices(stack_len,
                                                                                slice_len,
                                                                                slice_overlap)

    expected_start_indices = np.arange(0, stack_len, slice_len)
    assert np.all(np.equal(slice_start_indices, expected_start_indices))

    # test overlapping slices
    stack_len = 40
    slice_len = 4
    slice_overlap = 1
    slice_start_indices, slice_end_indices = reshape_data.compute_slice_indices(stack_len,
                                                                                slice_len,
                                                                                slice_overlap)

    assert len(slice_start_indices) == int(np.floor(stack_len / (slice_len - slice_overlap)))
    assert slice_end_indices[-1] == stack_len
    assert slice_end_indices[0] - slice_start_indices[0] == slice_len


def test_slice_helper():
    # test output shape with even division of slice
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 4

    slice_start_indices, slice_end_indices = reshape_data.compute_slice_indices(stack_len,
                                                                                slice_stack_len, 0)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num, row_len=row_len, col_len=col_len,
                                chan_len=chan_len)

    slice_output = reshape_data.slice_helper(input_data, slice_start_indices, slice_end_indices)

    assert slice_output.shape == (fov_len, slice_stack_len, crop_num,
                                  int(np.ceil(stack_len / slice_stack_len)),
                                  row_len, col_len, chan_len)

    # test output shape with uneven division of slice
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 6

    slice_start_indices, slice_end_indices = reshape_data.compute_slice_indices(stack_len,
                                                                                slice_stack_len, 0)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num, row_len=row_len, col_len=col_len,
                                chan_len=chan_len)

    slice_output = reshape_data.slice_helper(input_data, slice_start_indices, slice_end_indices)

    assert slice_output.shape == (fov_len, slice_stack_len, crop_num,
                                  (np.ceil(stack_len / slice_stack_len)),
                                  row_len, col_len, chan_len)

    # test output shape with slice overlaps
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 6
    slice_overlap = 1
    slice_start_indices, slice_end_indices = reshape_data.compute_slice_indices(stack_len,
                                                                                slice_stack_len,
                                                                                slice_overlap)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num, row_len=row_len, col_len=col_len,
                                chan_len=chan_len)

    slice_output = reshape_data.slice_helper(input_data, slice_start_indices, slice_end_indices)

    assert slice_output.shape == (fov_len, slice_stack_len, crop_num,
                                  (np.ceil(stack_len / (slice_stack_len - slice_overlap))),
                                  row_len, col_len, chan_len)

    # test output values
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 4
    slice_start_indices, slice_end_indices = reshape_data.compute_slice_indices(stack_len,
                                                                                slice_stack_len, 0)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num, row_len=row_len, col_len=col_len,
                                chan_len=chan_len)

    # tag upper left hand corner of each image
    tags = np.arange(stack_len)
    input_data[0, :, 0, 0, 0, 0, 0] = tags

    slice_output = reshape_data.slice_helper(input_data, slice_start_indices, slice_end_indices)

    # loop through each slice, make sure values increment as expected
    for i in range(slice_output.shape[1]):
        assert np.all(np.equal(slice_output[0, :, 0, i, 0, 0, 0], tags[i * 4:(i + 1) * 4]))


def test_create_slice_data():
    # test output shape with even division of slice
    fov_len, stack_len, num_crops, num_slices, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 4

    X_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=num_crops,
                            slice_num=num_slices, row_len=row_len, col_len=col_len,
                            chan_len=chan_len)

    y_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=num_crops,
                            slice_num=num_slices, row_len=row_len, col_len=col_len,
                            chan_len=chan_len)

    X_slice, y_slice, slice_indices = reshape_data.create_slice_data(X_data, y_data, slice_stack_len)

    assert X_slice.shape == (fov_len, slice_stack_len, num_crops,
                              int(np.ceil(stack_len / slice_stack_len)),
                              row_len, col_len, chan_len)


def test_save_npzs_for_caliban():
    fov_len, stack_len, num_crops, num_slices, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 4

    X_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=num_crops,
                                slice_num=num_slices,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    y_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=num_crops,
                            slice_num=num_slices,
                            row_len=row_len, col_len=col_len, chan_len=1)

    sliced_X, sliced_y, log_data = reshape_data.create_slice_data(X_data=X_data, y_data=y_data,
                                                                  slice_stack_len=slice_stack_len)

    with tempfile.TemporaryDirectory() as temp_dir:
        reshape_data.save_npzs_for_caliban(X_data=sliced_X, y_data=sliced_y, original_data=X_data,
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

        reshape_data.save_npzs_for_caliban(X_data=X_cropped, y_data=y_cropped,
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
        reshape_data.save_npzs_for_caliban(X_data=sliced_X, y_data=sliced_y,
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
        reshape_data.save_npzs_for_caliban(X_data=sliced_X, y_data=sliced_y,
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
        reshape_data.save_npzs_for_caliban(X_data=sliced_X, y_data=sliced_y,
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
    output_string = reshape_data.get_saved_file_path(dir_list, fov, crop, slice)

    assert output_string == dir_list[0]

    # test single modified npz
    slice = 5
    output_string = reshape_data.get_saved_file_path(dir_list, fov, crop, slice)
    assert output_string == dir_list[1]

    # test that error is raised when multiple save versions present
    slice = 6
    with pytest.raises(ValueError):
        output_string = reshape_data.get_saved_file_path(dir_list, fov, crop, slice)

    # test that error is raised when multiple save versions present due to resaves
    slice = 7

    with pytest.raises(ValueError):
        output_string = reshape_data.get_saved_file_path(dir_list, fov, crop, slice)


def test_load_npzs():
    with tempfile.TemporaryDirectory() as temp_dir:
        # first generate image stack that will be sliced up
        fov_len, stack_len, crop_num, slice_num = 1, 40, 1, 1
        row_len, col_len, chan_len = 50, 50, 3
        slice_stack_len = 4

        input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                    slice_num=slice_num,
                                    row_len=row_len, col_len=col_len, chan_len=chan_len)

        # slice the data
        slice_xr, log_data = reshape_data.create_slice_data(input_data, slice_stack_len)

        # crop the data
        crop_size = (10, 10)
        overlap_frac = 0.2
        data_xr_cropped, log_data_crop = \
            reshape_data.crop_multichannel_data(
                data_xr=slice_xr,
                crop_size=crop_size,
                overlap_frac=overlap_frac,
                test_parameters=False)
        # tag the upper left hand corner of the label in each slice
        slice_tags = np.arange(data_xr_cropped.shape[3])
        crop_tags = np.arange(data_xr_cropped.shape[2])
        data_xr_cropped[0, 0, :, 0, 0, 0, 2] = crop_tags
        data_xr_cropped[0, 0, 0, :, 0, 0, 2] = slice_tags

        combined_log_data = {**log_data, **log_data_crop}

        # save the tagged data
        reshape_data.save_npzs_for_caliban(resized_xr=data_xr_cropped, original_xr=input_data,
                                           log_data=combined_log_data, save_dir=temp_dir,
                                           blank_labels="include", save_format="npz",
                                           verbose=False)

        with open(os.path.join(temp_dir, "log_data.json")) as json_file:
            saved_log_data = json.load(json_file)

        loaded_slices = reshape_data.load_npzs(temp_dir, saved_log_data, verbose=False)

        # dims other than channels are the same
        assert (np.all(loaded_slices.shape[:-1] == data_xr_cropped.shape[:-1]))

        assert np.all(np.equal(loaded_slices[0, 0, :, 0, 0, 0, 0], crop_tags))
        assert np.all(np.equal(loaded_slices[0, 0, 0, :, 0, 0, 0], slice_tags))

    # test slices with unequal last length
    with tempfile.TemporaryDirectory() as temp_dir:
        # first generate image stack that will be sliced up
        fov_len, stack_len, crop_num, slice_num = 1, 40, 1, 1
        row_len, col_len, chan_len = 50, 50, 3
        slice_stack_len = 7

        input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                    slice_num=slice_num,
                                    row_len=row_len, col_len=col_len, chan_len=chan_len)

        # slice the data
        slice_xr, log_data = reshape_data.create_slice_data(input_data, slice_stack_len)

        # crop the data
        crop_size = (10, 10)
        overlap_frac = 0.2
        data_xr_cropped, log_data_crop = \
            reshape_data.crop_multichannel_data(
                data_xr=slice_xr,
                crop_size=crop_size,
                overlap_frac=overlap_frac,
                test_parameters=False)

        # tag the upper left hand corner of the annotations in each slice
        slice_tags = np.arange(data_xr_cropped.shape[3])
        crop_tags = np.arange(data_xr_cropped.shape[2])
        data_xr_cropped[0, 0, :, 0, 0, 0, 2] = crop_tags
        data_xr_cropped[0, 0, 0, :, 0, 0, 2] = slice_tags

        combined_log_data = {**log_data, **log_data_crop}

        # save the tagged data
        reshape_data.save_npzs_for_caliban(resized_xr=data_xr_cropped, original_xr=input_data,
                                           log_data=combined_log_data, save_dir=temp_dir,
                                           blank_labels="include", save_format="npz",
                                           verbose=False)

        loaded_slices = reshape_data.load_npzs(temp_dir, combined_log_data)

        # dims other than channels are the same
        assert (np.all(loaded_slices.shape[:-1] == data_xr_cropped.shape[:-1]))

        assert np.all(np.equal(loaded_slices[0, 0, :, 0, 0, 0, 0], crop_tags))
        assert np.all(np.equal(loaded_slices[0, 0, 0, :, 0, 0, 0], slice_tags))


def test_stitch_crops():
    # generate stack of crops from image with grid pattern
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 2, 1, 1, 1, 400, 400, 4

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    # create image with artificial objects to be segmented

    cell_idx = 1
    for i in range(12):
        for j in range(11):
            for fov in range(input_data.shape[0]):
                input_data[fov, :, :, :, (i * 35):(i * 35 + 10 + fov * 10),
                           (j * 37):(j * 37 + 8 + fov * 10), 3] = cell_idx
            cell_idx += 1

    # crop the image
    crop_size, overlap_frac = 400, 0.2
    cropped, log_data = reshape_data.crop_multichannel_data(data_xr=input_data,
                                                            crop_size=(crop_size, crop_size),
                                                            overlap_frac=overlap_frac)
    cropped_labels = cropped[..., -1:].values
    log_data["original_shape"] = input_data.shape

    # stitch the crops back together
    stitched_img = reshape_data.stitch_crops(annotated_data=cropped_labels, log_data=log_data)

    # trim padding
    row_padding, col_padding = log_data["row_padding"], log_data["col_padding"]
    if row_padding > 0:
        stitched_img = stitched_img[:, :, :, :, :-row_padding, :, :]
    if col_padding > 0:
        stitched_img = stitched_img[:, :, :, :, :, :-col_padding, :]

    # dims other than channels are the same
    assert np.all(stitched_img.shape[:-1] == input_data.shape[:-1])

    # check that objects are at same location
    assert (np.all(np.equal(stitched_img[..., 0] > 0, input_data.values[..., 3] > 0)))

    # check that same number of unique objects
    assert len(np.unique(stitched_img)) == len(np.unique(input_data.values))

    # test stitching imperfect annotator labels that slightly overlap
    # generate stack of crops from image with grid pattern
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 1, 1, 1, 800, 800, 1

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)
    side_len = 40
    cell_num = input_data.shape[4] // side_len

    cell_id = np.arange(1, cell_num ** 2 + 1)
    cell_id = np.random.choice(cell_id, cell_num ** 2, replace=False)
    cell_idx = 0
    for row in range(cell_num):
        for col in range(cell_num):
            input_data[0, 0, 0, 0, row * side_len:(row + 1) * side_len,
                       col * side_len:(col + 1) * side_len, 0] = cell_id[cell_idx]
            cell_idx += 1

    crop_size, overlap_frac = 100, 0.2

    starts, ends, padding = reshape_data.compute_crop_indices(img_len=row_len, crop_size=crop_size,
                                                              overlap_frac=overlap_frac)

    # generate a vector of random offsets to jitter the crop window,
    # simulating mismatches between frames
    offset_len = 5
    row_offset = np.append(
        np.append(0, np.random.randint(-offset_len, offset_len, len(starts) - 2)), 0)
    col_offset = np.append(
        np.append(0, np.random.randint(-offset_len, offset_len, len(starts) - 2)), 0)

    # modify indices by random offset
    row_starts, row_ends = starts + row_offset, ends + row_offset
    col_starts, col_ends = starts + col_offset, ends + col_offset

    cropped, padded = reshape_data.crop_helper(input_data=input_data, row_starts=row_starts,
                                               row_ends=row_ends,
                                               col_starts=col_starts, col_ends=col_ends,
                                               padding=(padding, padding))

    # generate log data, since we had to go inside the upper level
    # function to modify crop_helper inputs
    log_data = {}
    log_data["row_starts"] = row_starts.tolist()
    log_data["row_ends"] = row_ends.tolist()
    log_data["row_crop_size"] = crop_size
    log_data["num_row_crops"] = len(row_starts)
    log_data["col_starts"] = col_starts.tolist()
    log_data["col_ends"] = col_ends.tolist()
    log_data["col_crop_size"] = crop_size
    log_data["num_col_crops"] = len(col_starts)
    log_data["row_padding"] = int(padding)
    log_data["col_padding"] = int(padding)
    log_data["num_crops"] = cropped.shape[2]
    log_data["original_shape"] = input_data.shape
    log_data["fov_names"] = input_data.fovs.values.tolist()
    log_data["channel_names"] = input_data.channels.values.tolist()

    cropped_labels = cropped[..., -1:].values

    stitched_img = reshape_data.stitch_crops(annotated_data=cropped_labels, log_data=log_data)

    # trim padding
    stitched_img = stitched_img[:, :, :, :, :-padding, :-padding, :]

    relabeled = skimage.measure.label(stitched_img[0, 0, 0, 0, :, :, 0])

    props = skimage.measure.regionprops_table(relabeled, properties=["area", "label"])

    # dims other than channels are the same
    assert np.all(stitched_img.shape[:-1] == input_data.shape[:-1])

    # same number of unique objects before and after
    assert (len(np.unique(relabeled)) == len(np.unique(input_data[0, 0, 0, 0, :, :, 0])))

    # no cell is smaller than offset subtracted from each side
    min_size = (side_len - offset_len * 2) ** 2
    max_size = (side_len + offset_len * 2) ** 2

    assert (np.all(props["area"] <= max_size))
    assert (np.all(props["area"] >= min_size))


def test_reconstruct_image_data():
    # generate stack of crops from image with grid pattern
    with tempfile.TemporaryDirectory() as temp_dir:
        fov_len, stack_len, crop_num, slice_num = 2, 1, 1, 1
        row_len, col_len, chan_len = 400, 400, 4

        input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                    slice_num=slice_num,
                                    row_len=row_len, col_len=col_len, chan_len=chan_len)

        # create image with
        cell_idx = 1
        for i in range(12):
            for j in range(11):
                for fov in range(input_data.shape[0]):
                    input_data[fov, :, :, :, (i * 35):(i * 35 + 10 + fov * 10),
                               (j * 37):(j * 37 + 8 + fov * 10), 3] = cell_idx
                cell_idx += 1

        crop_size, overlap_frac = 40, 0.2

        # crop data
        data_xr_cropped, log_data = \
            reshape_data.crop_multichannel_data(data_xr=input_data,
                                                crop_size=(crop_size, crop_size),
                                                overlap_frac=0.2)

        # stitch data
        reshape_data.save_npzs_for_caliban(resized_xr=data_xr_cropped, original_xr=input_data,
                                           log_data=log_data,
                                           save_dir=temp_dir, verbose=False)

        reshape_data.reconstruct_image_stack(crop_dir=temp_dir)

        stitched_xr = xr.open_dataarray(os.path.join(temp_dir, "stitched_images.nc"))

        # dims other than channels are the same
        assert np.all(stitched_xr.shape[:-1] == input_data.shape[:-1])

        # all the same pixels are marked
        assert (np.all(np.equal(stitched_xr[:, :, 0] > 0, input_data[:, :, 0] > 0)))

        # there are the same number of cells
        assert (len(np.unique(stitched_xr)) == len(np.unique(input_data)))

    with tempfile.TemporaryDirectory() as temp_dir:
        # test single crop in x
        crop_size, overlap_frac = (400, 40), 0.2

        # crop data
        data_xr_cropped, log_data = reshape_data.crop_multichannel_data(data_xr=input_data,
                                                                        crop_size=crop_size,
                                                                        overlap_frac=0.2)

        # stitch data
        reshape_data.save_npzs_for_caliban(resized_xr=data_xr_cropped, original_xr=input_data,
                                           log_data=log_data,
                                           save_dir=temp_dir, verbose=False)

        reshape_data.reconstruct_image_stack(crop_dir=temp_dir)

        stitched_xr = xr.open_dataarray(os.path.join(temp_dir, "stitched_images.nc"))

        # dims other than channels are the same
        assert np.all(stitched_xr.shape[:-1] == input_data.shape[:-1])

        # all the same pixels are marked
        assert (np.all(np.equal(stitched_xr[:, :, 0] > 0, input_data[:, :, 0] > 0)))

        # there are the same number of cells
        assert (len(np.unique(stitched_xr)) == len(np.unique(input_data)))

    with tempfile.TemporaryDirectory() as temp_dir:
        # test single crop in both
        crop_size, overlap_frac = (400, 400), 0.2

        # crop data
        data_xr_cropped, log_data = reshape_data.crop_multichannel_data(data_xr=input_data,
                                                                        crop_size=crop_size,
                                                                        overlap_frac=0.2)

        # stitch data
        reshape_data.save_npzs_for_caliban(resized_xr=data_xr_cropped, original_xr=input_data,
                                           log_data=log_data,
                                           save_dir=temp_dir, verbose=False)

        reshape_data.reconstruct_image_stack(crop_dir=temp_dir)

        stitched_xr = xr.open_dataarray(os.path.join(temp_dir, "stitched_images.nc"))

        # dims other than channels are the same
        assert np.all(stitched_xr.shape[:-1] == input_data.shape[:-1])

        # all the same pixels are marked
        assert (np.all(np.equal(stitched_xr[:, :, 0] > 0, input_data[:, :, 0] > 0)))

        # there are the same number of cells
        assert (len(np.unique(stitched_xr)) == len(np.unique(input_data)))


def test_stitch_slices():
    # generate data
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 2, 1, 1, 1, 400, 400, 4

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num, row_len=row_len, col_len=col_len,
                                chan_len=chan_len)

    # create image with
    cell_idx = 1
    for i in range(12):
        for j in range(11):
            for fov in range(input_data.shape[0]):
                input_data[fov, :, :, :, (i * 35):(i * 35 + 10 + fov * 10),
                           (j * 37):(j * 37 + 8 + fov * 10), 3] = cell_idx
            cell_idx += 1

    crop_size, overlap_frac = 50, 0.2
    save_dir = "tests/caliban_toolbox/test_crop_and_stitch"

    # crop data
    data_xr_cropped, log_data = \
        reshape_data.crop_multichannel_data(data_xr=input_data,
                                            crop_size=(crop_size, crop_size),
                                            overlap_frac=0.2)

    # stitch data
    reshape_data.save_npzs_for_caliban(resized_xr=data_xr_cropped, original_xr=input_data,
                                       log_data=log_data,
                                       save_dir=save_dir)

    reshape_data.reconstruct_image_stack(crop_dir=save_dir)

    stitched_xr = xr.open_dataarray(os.path.join(save_dir, "stitched_images.nc"))

    # all the same pixels are marked
    assert (np.all(np.equal(stitched_xr[:, :, 0] > 0, input_data[:, :, 0] > 0)))

    # there are the same number of cells
    assert (len(np.unique(stitched_xr)) == len(np.unique(input_data)))

    # clean up
    shutil.rmtree(save_dir)


def test_stitch_slices():
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 4

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    # generate ordered data
    linear_seq = np.arange(stack_len * row_len * col_len)
    test_vals = linear_seq.reshape((stack_len, row_len, col_len))
    input_data[0, :, 0, 0, :, :, 2] = test_vals

    slice_xr, log_data = reshape_data.create_slice_data(input_data, slice_stack_len)

    # TODO move crop + slice testing to another test function
    crop_size = (10, 10)
    overlap_frac = 0.2
    data_xr_cropped, log_data_crop = reshape_data.crop_multichannel_data(data_xr=slice_xr,
                                                                         crop_size=crop_size,
                                                                         overlap_frac=overlap_frac,
                                                                         test_parameters=False)

    # # get parameters
    # row_crop_size, col_crop_size = crop_size[0], crop_size[1]
    # num_row_crops, num_col_crops = log_data_crop["num_row_crops"], log_data_crop["num_col_crops"]
    # num_slices = log_data["num_slices"]

    log_data["original_shape"] = input_data.shape
    log_data["fov_names"] = input_data.fovs.values
    stitched_slices = reshape_data.stitch_slices(slice_xr[..., -1:], {**log_data})

    # dims other than channels are the same
    assert np.all(stitched_slices.shape[:-1] == input_data.shape[:-1])

    assert np.all(np.equal(stitched_slices[0, :, 0, 0, :, :, 0], test_vals))

    # test case without even division of crops into imsize

    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 7

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    # generate ordered data
    linear_seq = np.arange(stack_len * row_len * col_len)
    test_vals = linear_seq.reshape((stack_len, row_len, col_len))
    input_data[0, :, 0, 0, :, :, 2] = test_vals

    slice_xr, log_data = reshape_data.create_slice_data(input_data, slice_stack_len)

    # get parameters
    log_data["original_shape"] = input_data.shape
    log_data["fov_names"] = input_data.fovs.values
    stitched_slices = reshape_data.stitch_slices(slice_xr[..., -1:], log_data)

    # dims other than channels are the same
    assert np.all(stitched_slices.shape[:-1] == input_data.shape[:-1])

    assert np.all(np.equal(stitched_slices[0, :, 0, 0, :, :, 0], test_vals))


def test_reconstruct_slice_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        # generate data
        fov_len, stack_len, crop_num, slice_num = 1, 40, 1, 1
        row_len, col_len, chan_len = 50, 50, 3
        slice_stack_len = 4

        input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                    slice_num=slice_num,
                                    row_len=row_len, col_len=col_len, chan_len=chan_len)

        # tag upper left hand corner of the label in each image
        tags = np.arange(stack_len)
        input_data[0, :, 0, 0, 0, 0, 2] = tags

        slice_xr, slice_log_data = reshape_data.create_slice_data(input_data, slice_stack_len)

        reshape_data.save_npzs_for_caliban(resized_xr=slice_xr, original_xr=input_data,
                                           log_data={**slice_log_data}, save_dir=temp_dir,
                                           blank_labels="include",
                                           save_format="npz", verbose=False)

        stitched_slices = reshape_data.reconstruct_slice_data(temp_dir)

        # dims other than channels are the same
        assert np.all(stitched_slices.shape[:-1] == input_data.shape[:-1])

        assert np.all(np.equal(stitched_slices[0, :, 0, 0, 0, 0, 0], tags))
