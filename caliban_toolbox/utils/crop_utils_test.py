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
import skimage

import numpy as np
from caliban_toolbox import reshape_data

from caliban_toolbox.utils import crop_utils
import xarray as xr


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
    starts, ends, padding = crop_utils.compute_crop_indices(img_len=img_len, crop_size=crop_size,
                                                              overlap_frac=overlap_frac)
    assert (len(starts) == 1)
    assert (len(ends) == 1)

    # test crop size that doesn't divide evenly into image size
    img_len, crop_size, overlap_frac = 105, 20, 0.2
    starts, ends, padding = crop_utils.compute_crop_indices(img_len=img_len, crop_size=crop_size,
                                                              overlap_frac=overlap_frac)
    crop_num = np.ceil(img_len / (crop_size - (crop_size * overlap_frac)))
    assert (len(starts) == crop_num)
    assert (len(ends) == crop_num)

    crop_end = crop_num * (crop_size - (crop_size * overlap_frac)) + crop_size * overlap_frac
    assert (ends[-1] == crop_end)

    # test overlap of 0 between crops
    img_len, crop_size, overlap_frac = 200, 20, 0
    starts, ends, padding = crop_utils.compute_crop_indices(img_len=img_len, crop_size=crop_size,
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

    starts, ends, padding = crop_utils.compute_crop_indices(img_len=row_len, crop_size=crop_size,
                                                              overlap_frac=overlap_frac)
    cropped, padded = crop_utils.crop_helper(input_data=test_xr, row_starts=starts,
                                               row_ends=ends, col_starts=starts, col_ends=ends,
                                               padding=(padding, padding))

    assert (cropped.shape == (fov_len, stack_len, 1, slice_num, row_len, col_len, chan_len))

    # test crops of different row/col dimensions
    row_crop, col_crop = 50, 40
    row_starts, row_ends, row_padding = \
        crop_utils.compute_crop_indices(img_len=row_len, crop_size=row_crop,
                                          overlap_frac=overlap_frac)

    col_starts, col_ends, col_padding = \
        crop_utils.compute_crop_indices(img_len=col_len, crop_size=col_crop,
                                          overlap_frac=overlap_frac)

    cropped, padded = crop_utils.crop_helper(input_data=test_xr, row_starts=row_starts,
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
        crop_utils.compute_crop_indices(img_len=row_len, crop_size=row_crop,
                                          overlap_frac=overlap_frac)

    col_starts, col_ends, col_padding = \
        crop_utils.compute_crop_indices(img_len=col_len, crop_size=col_crop,
                                          overlap_frac=overlap_frac)

    cropped, padded = crop_utils.crop_helper(input_data=test_xr, row_starts=row_starts,
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


def test_stitch_crops():
    # generate stack of crops from image with grid pattern
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 2, 1, 1, 1, 400, 400, 4

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

    # ## Test when crop is same size as image
    crop_size, overlap_frac = 400, 0.2
    X_cropped, y_cropped, log_data = \
        reshape_data.crop_multichannel_data(X_data=X_data,
                                            y_data=y_data,
                                            crop_size=(crop_size, crop_size),
                                            overlap_frac=overlap_frac)

    log_data["original_shape"] = X_data.shape

    # stitch the crops back together
    stitched_img = crop_utils.stitch_crops(crop_stack=y_cropped, log_data=log_data)

    # dims are the same
    assert np.all(stitched_img.shape == y_data.shape)

    # check that objects are at same location
    assert (np.all(np.equal(stitched_img[..., 0] > 0, y_data.values[..., 0] > 0)))

    # check that same number of unique objects
    assert len(np.unique(stitched_img)) == len(np.unique(y_data.values))

    # ## Test when rows has only one crop
    crop_size, overlap_frac = (400, 40), 0.2

    # crop data
    X_cropped, y_cropped, log_data = \
        reshape_data.crop_multichannel_data(X_data=X_data,
                                            y_data=y_data,
                                            crop_size=crop_size,
                                            overlap_frac=overlap_frac)

    # stitch back together
    log_data["original_shape"] = X_data.shape
    stitched_imgs = crop_utils.stitch_crops(crop_stack=y_cropped, log_data=log_data)

    # dims are the same
    assert np.all(stitched_imgs.shape == y_data.shape)

    # all the same pixels are marked
    assert (np.all(np.equal(stitched_imgs[:, :, 0] > 0, y_data[:, :, 0] > 0)))

    # there are the same number of cells
    assert (len(np.unique(stitched_imgs)) == len(np.unique(y_data)))

    # test stitching imperfect annotator labels that slightly overlap
    # generate stack of crops from image with grid pattern
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 1, 1, 1, 800, 800, 1

    y_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num,
                            slice_num=slice_num,
                            row_len=row_len, col_len=col_len, chan_len=chan_len)
    side_len = 40
    cell_num = y_data.shape[4] // side_len

    cell_id = np.arange(1, cell_num ** 2 + 1)
    cell_id = np.random.choice(cell_id, cell_num ** 2, replace=False)
    cell_idx = 0
    for row in range(cell_num):
        for col in range(cell_num):
            y_data[0, 0, 0, 0, row * side_len:(row + 1) * side_len,
                       col * side_len:(col + 1) * side_len, 0] = cell_id[cell_idx]
            cell_idx += 1

    crop_size, overlap_frac = 100, 0.2

    starts, ends, padding = crop_utils.compute_crop_indices(img_len=row_len, crop_size=crop_size,
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

    y_cropped, padded = crop_utils.crop_helper(input_data=y_data, row_starts=row_starts,
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
    log_data["num_crops"] = y_cropped.shape[2]
    log_data["original_shape"] = y_data.shape
    log_data["fov_names"] = y_data.fovs.values.tolist()
    log_data["channel_names"] = y_data.channels.values.tolist()

    stitched_img = crop_utils.stitch_crops(crop_stack=y_cropped, log_data=log_data)

    relabeled = skimage.measure.label(stitched_img[0, 0, 0, 0, :, :, 0])

    props = skimage.measure.regionprops_table(relabeled, properties=["area", "label"])

    # dims are the same
    assert np.all(stitched_img.shape == y_data.shape)

    # same number of unique objects before and after
    assert (len(np.unique(relabeled)) == len(np.unique(y_data[0, 0, 0, 0, :, :, 0])))

    # no cell is smaller than offset subtracted from each side
    min_size = (side_len - offset_len * 2) ** 2
    max_size = (side_len + offset_len * 2) ** 2

    assert (np.all(props["area"] <= max_size))
    assert (np.all(props["area"] >= min_size))