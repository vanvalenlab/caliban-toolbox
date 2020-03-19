import os
import shutil
import json
import copy

import numpy as np
from caliban_toolbox.pre_annotation import npz_preprocessing
import xarray as xr

import importlib
importlib.reload(npz_preprocessing)


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

    test_stack_xr = xr.DataArray(data=test_img, coords=[fovs, range(stack_len), range(crop_num), range(slice_num),
                                                        range(row_len), range(col_len), channels],
                           dims=["fovs", "stacks", "crops", "slices", "rows", "cols", "channels"])

    return test_stack_xr


def test_compute_crop_indices():
    # test corner case of only one crop
    img_len, crop_size, overlap_frac = 100, 100, 0.2
    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=img_len, crop_size=crop_size,
                                                                   overlap_frac=overlap_frac)
    assert(len(starts) == 1)
    assert(len(ends) == 1)

    # test crop size that doesn't divide evenly into image size
    img_len, crop_size, overlap_frac = 105, 20, 0.2
    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=img_len, crop_size=crop_size,
                                                                   overlap_frac=overlap_frac)
    crop_num = np.ceil(img_len / (crop_size - (crop_size * overlap_frac)))
    assert(len(starts) == crop_num)
    assert(len(ends) == crop_num)

    crop_end = crop_num * (crop_size - (crop_size * overlap_frac)) + crop_size * overlap_frac
    assert(ends[-1] == crop_end)

    # test overlap of 0 between crops
    img_len, crop_size, overlap_frac = 200, 20, 0
    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=img_len, crop_size=crop_size,
                                                                   overlap_frac=overlap_frac)
    assert (np.all(starts == range(0, 200, 20)))
    assert (np.all(ends == range(20, 201, 20)))
    assert (padding == 0)


def test_crop_helper():

    # img params
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 2, 1, 1, 1, 200, 200, 1
    crop_size, overlap_frac = 200, 0.2

    # test only one crop
    test_xr = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=slice_num,
                             row_len=row_len, col_len=col_len, chan_len=chan_len)

    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=row_len, crop_size=crop_size,
                                                                               overlap_frac=overlap_frac)
    cropped, padded = npz_preprocessing.crop_helper(input_data=test_xr, row_start=starts, row_end=ends,
                                                    col_start=starts, col_end=ends,
                                                    padding=(padding, padding))

    assert(cropped.shape == (fov_len, stack_len, 1, slice_num, row_len, col_len, chan_len))

    # test crops of different row/col dimensions
    row_crop, col_crop = 50, 40
    row_starts, row_ends, row_padding = npz_preprocessing.compute_crop_indices(img_len=row_len, crop_size=row_crop,
                                                                               overlap_frac=overlap_frac)

    col_starts, col_ends, col_padding = npz_preprocessing.compute_crop_indices(img_len=col_len, crop_size=col_crop,
                                                                               overlap_frac=overlap_frac)

    cropped, padded = npz_preprocessing.crop_helper(input_data=test_xr, row_start=row_starts, row_end=row_ends,
                                                    col_start=col_starts, col_end=col_ends,
                                                    padding=(row_padding, col_padding))

    assert(cropped.shape == (fov_len, stack_len, 30, slice_num, row_crop, col_crop, chan_len))

    # test that correct region of image is being cropped
    row_crop, col_crop = 40, 40

    # assign each pixel in the image a unique value
    linear_sequence = np.arange(0, fov_len * 1 * 1 * row_len * col_len * chan_len)
    linear_sequence_reshaped = np.reshape(linear_sequence, (fov_len, 1, 1, 1, row_len, col_len, chan_len))
    test_xr[:, :, :, :, :, :, :] = linear_sequence_reshaped

    # crop the image
    row_starts, row_ends, row_padding = npz_preprocessing.compute_crop_indices(img_len=row_len, crop_size=row_crop,
                                                                               overlap_frac=overlap_frac)

    col_starts, col_ends, col_padding = npz_preprocessing.compute_crop_indices(img_len=col_len, crop_size=col_crop,
                                                                               overlap_frac=overlap_frac)

    cropped, padded = npz_preprocessing.crop_helper(input_data=test_xr, row_start=row_starts, row_end=row_ends,
                                                    col_start=col_starts, col_end=col_ends,
                                                    padding=(row_padding, col_padding))

    # check that the values of each crop match the value in uncropped image
    for img in range(test_xr.shape[0]):
        crop_counter = 0
        for row in range(len(row_starts)):
            for col in range(len(col_starts)):
                crop = cropped[img, 0, crop_counter, 0, :, :, 0].values

                original_image_crop = test_xr[img, 0, 0, 0, row_ends[row]:row_ends[row], col_starts[col]:col_ends[col], :].values
                assert(np.all(crop == original_image_crop))

                crop_counter += 1


def test_crop_multichannel_data():
    # img params
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 2, 1, 1, 1, 200, 200, 1
    crop_size = (50, 50)
    overlap_frac = 0.2

    # test only one crop
    test_xr = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=slice_num,
                             row_len=row_len, col_len=col_len, chan_len=chan_len)

    data_xr_cropped, log_data = npz_preprocessing.crop_multichannel_data(data_xr=test_xr, crop_size=crop_size,
                                                                         overlap_frac=overlap_frac, test_parameters=False)



    expected_crop_num = len(npz_preprocessing.compute_crop_indices(row_len, crop_size[0], overlap_frac)[0]) ** 2
    assert (data_xr_cropped.shape == (fov_len, stack_len, expected_crop_num, slice_num, crop_size[0], crop_size[1], chan_len))

    assert log_data["num_crops"] == expected_crop_num


def test_compute_montage_indices():

    # test when montage divides evenly into stack len
    slice_len = 40
    montage_len = 4
    montage_overlap = 0
    montage_start_indices, montage_end_indices = npz_preprocessing.compute_montage_indices(slice_len, montage_len,
                                                                                           montage_overlap)
    assert np.all(np.equal(montage_start_indices, np.arange(0, slice_len, montage_len)))

    # test when montage_num does not divide evenly into slice_num
    slice_len = 42
    montage_len = 5
    montage_start_indices, montage_end_indices = npz_preprocessing.compute_montage_indices(slice_len, montage_len,
                                                                                           montage_overlap)

    expected_start_indices = np.arange(0, slice_len, montage_len)
    assert np.all(np.equal(montage_start_indices, expected_start_indices))

    # test overlapping montages
    slice_len = 40
    montage_len = 4
    montage_overlap = 1
    montage_start_indices, montage_end_indices = npz_preprocessing.compute_montage_indices(slice_len, montage_len,
                                                                                           montage_overlap)
    assert len(montage_start_indices) == int(np.floor(slice_len / (montage_len - montage_overlap)))
    assert montage_end_indices[-1] == slice_len
    assert montage_end_indices[0] - montage_start_indices[0] == montage_len


def test_montage_helper():

    # test output shape with even division of montage
    fov_len, stack_len, crop_num, montage_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    montage_stack_len = 4

    montage_start_indices, montage_end_indices = npz_preprocessing.compute_montage_indices(stack_len, montage_stack_len, 0)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=montage_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    montage_output = npz_preprocessing.montage_helper(input_data, montage_start_indices, montage_end_indices)

    assert montage_output.shape == (fov_len, montage_stack_len, crop_num, int(np.ceil(stack_len / montage_stack_len)),
                                    row_len, col_len, chan_len)

    # test output shape with uneven division of montage
    fov_len, stack_len, crop_num, montage_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    montage_stack_len = 6

    montage_start_indices, montage_end_indices = npz_preprocessing.compute_montage_indices(stack_len, montage_stack_len, 0)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=montage_num,
                                 row_len=row_len, col_len=col_len, chan_len=chan_len)

    montage_output = npz_preprocessing.montage_helper(input_data, montage_start_indices, montage_end_indices)

    assert montage_output.shape == (fov_len, montage_stack_len, crop_num, (np.ceil(stack_len / montage_stack_len)),
                                    row_len, col_len, chan_len)

    # test output shape with montage overlaps
    fov_len, stack_len, crop_num, montage_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    montage_stack_len = 6
    montage_overlap = 1
    montage_start_indices, montage_end_indices = npz_preprocessing.compute_montage_indices(stack_len, montage_stack_len,
                                                                                           montage_overlap)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=montage_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    montage_output = npz_preprocessing.montage_helper(input_data, montage_start_indices, montage_end_indices)

    assert montage_output.shape == (fov_len, montage_stack_len, crop_num,
                                    (np.ceil(stack_len / (montage_stack_len - montage_overlap))),
                                    row_len, col_len, chan_len)

    # test output values
    fov_len, stack_len, crop_num, montage_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    montage_stack_len = 4
    montage_start_indices, montage_end_indices = npz_preprocessing.compute_montage_indices(stack_len, montage_stack_len, 0)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=montage_num,
                                 row_len=row_len, col_len=col_len, chan_len=chan_len)

    # tag upper left hand corner of each image
    tags = np.arange(stack_len)
    input_data[0, :, 0, 0, 0, 0, 0] = tags

    montage_output = npz_preprocessing.montage_helper(input_data,  montage_start_indices, montage_end_indices)

    # loop through each montage, make sure values increment as expected
    for i in range(montage_output.shape[1]):
        assert np.all(np.equal(montage_output[0, :, 0, i, 0, 0, 0], tags[i * 4:(i + 1) * 4]))


def test_create_montage_data():

    # test output shape with even division of montage
    fov_len, stack_len, crop_num, montage_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    montage_stack_len = 4

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=montage_num,
                                 row_len=row_len, col_len=col_len, chan_len=chan_len)

    montage_xr, montage_indices = npz_preprocessing.create_montage_data(input_data, montage_stack_len)

    assert montage_xr.shape == (fov_len, montage_stack_len, crop_num, int(np.ceil(stack_len / montage_stack_len)),
                                row_len, col_len, chan_len)


def test_save_npzs_for_caliban():
    fov_len, stack_len, crop_num, montage_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    montage_stack_len = 4

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=montage_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    montage_xr, log_data = npz_preprocessing.create_montage_data(input_data, montage_stack_len)

    save_dir = "tests/caliban_toolbox/test_save_npzs_for_caliban/"
    npz_preprocessing.save_npzs_for_caliban(resized_xr=montage_xr, original_xr=input_data, log_data=copy.copy(log_data),
                                            save_dir=save_dir, blank_labels="include", save_format="npz")

    # check that correct size was saved

    test_npz_labels = np.load(save_dir + "fov_fov1_row_0_col_0_montage_0.npz")

    assert test_npz_labels["y"].shape == (fov_len, montage_stack_len, row_len, col_len, 1)

    assert test_npz_labels["y"].shape[:-1] == test_npz_labels["X"].shape[:-1]

    # check that json saved successfully
    with open(os.path.join(save_dir, "log_data.json")) as json_file:
        saved_log_data = json.load(json_file)

    assert saved_log_data["original_shape"] == list(input_data.shape)
    shutil.rmtree(save_dir)

    # check that combined crop and montage saving works
    crop_size = (10, 10)
    overlap_frac = 0.2
    data_xr_cropped, log_data_crop = npz_preprocessing.crop_multichannel_data(data_xr=montage_xr, crop_size=crop_size,
                                                                         overlap_frac=overlap_frac,
                                                                         test_parameters=False)

    npz_preprocessing.save_npzs_for_caliban(resized_xr=data_xr_cropped, original_xr=input_data,
                                            log_data={**log_data, **log_data_crop}, save_dir=save_dir,
                                            blank_labels="include", save_format="npz")
    expected_crop_num = data_xr_cropped.shape[2] * data_xr_cropped.shape[3]
    files = os.listdir(save_dir)
    files = [file for file in files if "npz" in file]

    assert len(files) == expected_crop_num
    shutil.rmtree(save_dir)

    # check that arguments specifying what to do with blank crops are working
    # set specified crops to not be blank
    montage_xr[0, 0, 0, [1, 4, 7], 0, 0, -1] = 27
    np.sum(np.nonzero(montage_xr.values))

    expected_crop_num = montage_xr.shape[2] * montage_xr.shape[3]

    # test that function correctly includes blank crops when saving
    npz_preprocessing.save_npzs_for_caliban(resized_xr=montage_xr, original_xr=input_data, log_data=copy.copy(log_data),
                                            save_dir=save_dir, blank_labels="include", save_format="npz")

    # check that there is the expected number of files saved to directory
    files = os.listdir(save_dir)
    files = [file for file in files if "npz" in file]

    assert len(files) == expected_crop_num
    shutil.rmtree(save_dir)

    # test that function correctly skips blank crops when saving
    npz_preprocessing.save_npzs_for_caliban(resized_xr=montage_xr, original_xr=input_data, log_data=copy.copy(log_data),
                                            save_dir=save_dir, save_format="npz", blank_labels="skip")

    #  check that expected number of files in directory
    files = os.listdir(save_dir)
    files = [file for file in files if "npz" in file]
    assert len(files) == 3
    shutil.rmtree(save_dir)

    # test that function correctly saves blank crops to separate folder
    npz_preprocessing.save_npzs_for_caliban(resized_xr=montage_xr, original_xr=input_data, log_data=copy.copy(log_data),
                                            save_dir=save_dir, save_format="npz", blank_labels="separate")

    # check that expected number of files in each directory
    files = os.listdir(save_dir)
    files = [file for file in files if "npz" in file]
    assert len(files) == 3

    files = os.listdir(os.path.join(save_dir, "separate"))
    files = [file for file in files if "npz" in file]
    assert len(files) == expected_crop_num - 3
    shutil.rmtree(save_dir)



