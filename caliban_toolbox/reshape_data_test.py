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
import xarray as xr

from caliban_toolbox import reshape_data
from caliban_toolbox.utils import crop_utils, io_utils
from caliban_toolbox.utils.crop_utils_test import _blank_data_xr


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

    expected_crop_num = len(crop_utils.compute_crop_indices(img_len=row_len,
                                                            crop_size=crop_size[0],
                                                            overlap_frac=overlap_frac)[0]) ** 2
    assert (X_data_cropped.shape == (fov_len, stack_len, expected_crop_num, slice_num,
                                     crop_size[0], crop_size[1], channel_len))

    assert log_data["num_crops"] == expected_crop_num


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

    X_slice, y_slice, slice_indices = reshape_data.create_slice_data(X_data, y_data,
                                                                     slice_stack_len)

    assert X_slice.shape == (fov_len, slice_stack_len, num_crops,
                             int(np.ceil(stack_len / slice_stack_len)),
                             row_len, col_len, chan_len)


def test_reconstruct_image_stack():
    with tempfile.TemporaryDirectory() as temp_dir:
        # generate stack of crops from image with grid pattern
        (fov_len, stack_len, crop_num,
         slice_num, row_len, col_len, chan_len) = 2, 1, 1, 1, 400, 400, 4

        X_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

        y_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=1)

        # create image with artificial objects to be segmented

        cell_idx = 1
        for i in range(12):
            for j in range(11):
                for fov in range(y_data.shape[0]):
                    y_data[fov, :, :, :, (i * 35):(i * 35 + 10 + fov * 10),
                           (j * 37):(j * 37 + 8 + fov * 10), 0] = cell_idx
                cell_idx += 1

        # Crop the data
        crop_size, overlap_frac = 100, 0.2
        X_cropped, y_cropped, log_data = \
            reshape_data.crop_multichannel_data(X_data=X_data,
                                                y_data=y_data,
                                                crop_size=(crop_size, crop_size),
                                                overlap_frac=overlap_frac)

        io_utils.save_npzs_for_caliban(X_data=X_cropped, y_data=y_cropped, original_data=X_data,
                                       log_data=log_data, save_dir=temp_dir)

        reshape_data.reconstruct_image_stack(crop_dir=temp_dir)

        stitched_imgs = xr.open_dataarray(os.path.join(temp_dir, 'stitched_images.xr'))

        # dims are the same
        assert np.all(stitched_imgs.shape == y_data.shape)

        # all the same pixels are marked
        assert (np.all(np.equal(stitched_imgs[:, :, 0] > 0, y_data[:, :, 0] > 0)))

        # there are the same number of cells
        assert (len(np.unique(stitched_imgs)) == len(np.unique(y_data)))

    with tempfile.TemporaryDirectory() as temp_dir:
        # generate data with the corner tagged
        fov_len, stack_len, crop_num, slice_num = 1, 40, 1, 1
        row_len, col_len, chan_len = 50, 50, 3
        slice_stack_len = 4

        X_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

        y_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=1)

        # tag upper left hand corner of the label in each image
        tags = np.arange(stack_len)
        y_data[0, :, 0, 0, 0, 0, 0] = tags

        X_slice, y_slice, slice_log_data = \
            reshape_data.create_slice_data(X_data=X_data,
                                           y_data=y_data,
                                           slice_stack_len=slice_stack_len)

        io_utils.save_npzs_for_caliban(X_data=X_slice, y_data=y_slice, original_data=X_data,
                                       log_data={**slice_log_data}, save_dir=temp_dir,
                                       blank_labels="include",
                                       save_format="npz", verbose=False)

        reshape_data.reconstruct_image_stack(temp_dir)
        stitched_imgs = xr.open_dataarray(os.path.join(temp_dir, 'stitched_images.xr'))

        assert np.all(stitched_imgs.shape == y_data.shape)
        assert np.all(np.equal(stitched_imgs[0, :, 0, 0, 0, 0, 0], tags))

    with tempfile.TemporaryDirectory() as temp_dir:
        # generate data with both corners tagged and images labeled

        (fov_len, stack_len, crop_num,
         slice_num, row_len, col_len, chan_len) = 1, 8, 1, 1, 400, 400, 4

        X_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

        y_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                                slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=1)

        # create image with artificial objects to be segmented

        cell_idx = 1
        for i in range(1, 12):
            for j in range(1, 11):
                for stack in range(stack_len):
                    y_data[:, stack, :, :, (i * 35):(i * 35 + 10 + stack * 2),
                           (j * 37):(j * 37 + 8 + stack * 2), 0] = cell_idx
                cell_idx += 1

        # tag upper left hand corner of each image with squares of increasing size
        for stack in range(stack_len):
            y_data[0, stack, 0, 0, :stack, :stack, 0] = 1

        # Crop the data
        crop_size, overlap_frac = 100, 0.2
        X_cropped, y_cropped, log_data = \
            reshape_data.crop_multichannel_data(X_data=X_data,
                                                y_data=y_data,
                                                crop_size=(crop_size, crop_size),
                                                overlap_frac=overlap_frac)

        X_slice, y_slice, slice_log_data = \
            reshape_data.create_slice_data(X_data=X_cropped,
                                           y_data=y_cropped,
                                           slice_stack_len=slice_stack_len)

        io_utils.save_npzs_for_caliban(X_data=X_slice, y_data=y_slice, original_data=X_data,
                                       log_data={**slice_log_data, **log_data},
                                       save_dir=temp_dir,
                                       blank_labels="include",
                                       save_format="npz", verbose=False)

        reshape_data.reconstruct_image_stack(temp_dir)
        stitched_imgs = xr.open_dataarray(os.path.join(temp_dir, 'stitched_images.xr'))

        assert np.all(stitched_imgs.shape == y_data.shape)

        # dims are the same
        assert np.all(stitched_imgs.shape == y_data.shape)

        # all the same pixels are marked
        assert (np.all(np.equal(stitched_imgs[:, :, 0] > 0, y_data[:, :, 0] > 0)))

        # there are the same number of cells
        assert (len(np.unique(stitched_imgs)) == len(np.unique(y_data)))

        # check mark in upper left hand corner of image
        for stack in range(stack_len):
            original = np.zeros((10, 10))
            original[:stack, :stack] = 1
            new = stitched_imgs[0, stack, 0, 0, :10, :10, 0]
            assert np.array_equal(original > 0, new > 0)
