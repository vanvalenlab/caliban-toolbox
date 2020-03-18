import os
import shutil
import json

import numpy as np
from caliban_toolbox.pre_annotation import npz_preprocessing
from caliban_toolbox.post_annotation import npz_postprocessing
import xarray as xr
import skimage.measure

import importlib
importlib.reload(npz_preprocessing)
importlib.reload(npz_postprocessing)


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


# tests for npz version of the pipeline
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


def test_save_crops():

    # create fake stack of crops
    fov_num, crop_num, row_len, col_len, chan_num = 2, 25, 50, 40, 1
    test_xr = _blank_cropped_xr(fov_num, crop_num, row_len, col_len, chan_num)

    # set specified crops to not be blank
    non_blank_crops = np.random.choice(range(int(crop_num)), size=15, replace=False)
    test_xr[:, non_blank_crops, :, 0, 0] = 27

    # save crops to folder
    base_dir = "tests/caliban_toolbox/"

    # test that function correctly includes blank crops when saving
    save_dir = os.path.join(base_dir, "test_crop_dir_all")
    os.makedirs(save_dir)
    npz_preprocessing.save_crops(cropped_data=test_xr, fov_names=test_xr.fovs.values, num_row_crops=5, num_col_crops=5,
                                 save_dir=save_dir, save_format="xr", blank_labels="include")

    # check that there is the expected number of files saved to directory
    files = os.listdir(save_dir)
    files = [file for file in files if "xr" in file]
    assert len(files) == fov_num * crop_num
    shutil.rmtree(save_dir)

    # test that function correctly skips blank crops when saving
    save_dir = os.path.join(base_dir, "test_crop_dir_non_blank")
    os.makedirs(save_dir)
    npz_preprocessing.save_crops(cropped_data=test_xr, fov_names=test_xr.fovs.values, num_row_crops=5, num_col_crops=5,
                                 save_dir=save_dir, save_format="xr", blank_labels="skip")

    #  check that expected number of files in directory
    files = os.listdir(save_dir)
    files = [file for file in files if "xr" in file]
    assert len(files) == fov_num * len(non_blank_crops)
    shutil.rmtree(save_dir)

    # test that function correctly saves blank crops to separate folder
    save_dir = os.path.join(base_dir, "test_crop_dir_separate")
    os.makedirs(save_dir)
    npz_preprocessing.save_crops(cropped_data=test_xr, fov_names=test_xr.fovs.values, num_row_crops=5, num_col_crops=5,
                                 save_dir=save_dir, save_format="xr", blank_labels="separate")

    # check that expected number of files in each directory
    files = os.listdir(save_dir)
    files = [file for file in files if "xr" in file]
    assert len(files) == fov_num * len(non_blank_crops)

    files = os.listdir(os.path.join(save_dir, "separate"))
    files = [file for file in files if "xr" in file]
    assert len(files) == fov_num * (crop_num - len(non_blank_crops))
    shutil.rmtree(save_dir)


def test_crop_multichannel_data():
    # img params
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 2, 1, 1, 1, 200, 200, 1
    crop_size = (50, 50)
    overlap_frac = 0.2

    # test only one crop
    test_xr = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=slice_num,
                             row_len=row_len, col_len=col_len, chan_len=chan_len)

    data_xr_cropped, log_data = npz_preprocessing.crop_multichannel_data(data_xr=test_xr, crop_size=crop_size,
                                                                         overlap_frac=overlap_frac, blank_labels="include",
                                                                         test_parameters=False)

    assert (np.all(log_data["fov_names"] == test_xr.fovs.values))

    expected_crop_num = len(npz_preprocessing.compute_crop_indices(row_len, crop_size[0], overlap_frac)[0]) ** 2
    assert (data_xr_cropped.shape == (fov_len, stack_len, expected_crop_num, slice_num, crop_size[0], crop_size[1], chan_len))


def test_stitch_crops():
    # generate stack of crops from image with grid pattern
    test_xr = _blank_xr(2, 400, 400, 4)

    cell_idx = 1
    for i in range(12):
        for j in range(11):
            for img in range(test_xr.shape[0]):
                test_xr[img, (i * 35):(i * 35 + 10 + img * 10), (j * 37):(j * 37 + 8 + img * 10), 3] = cell_idx
            cell_idx += 1

    # img params
    fov_num, row_num, col_num, chan_num = 2, 400, 400, 1
    crop_size, overlap_frac = 50, 0.2

    # test only one crop
    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=row_num, crop_size=crop_size,
                                                                               overlap_frac=overlap_frac)
    cropped, padded = npz_preprocessing.crop_helper(input_data=test_xr, row_start=starts, row_end=ends,
                                                    col_start=starts, col_end=ends,
                                                    padding=((0, 0), (0, padding), (0, padding), (0, 0)))
    cropped_labels = cropped[..., -1:].values

    stitched_img = npz_postprocessing.stitch_crops(stack=cropped_labels, padded_img_shape=padded, row_starts=starts,
                                                   row_ends=ends, col_starts=starts, col_ends=ends)

    # trim padding
    stitched_img = stitched_img[:, :-padding, :-padding, :]

    # check that objects are at same location
    assert(np.all(np.equal(stitched_img[:, :, :, 0] > 0, test_xr.values[:, :, :, 3] > 0)))

    # check that same number of unique objects
    assert(np.all(np.unique(stitched_img) == np.unique(test_xr.values)))


    # test stitching imperfect annotator labels that slightly overlap
    test_xr2 = _blank_xr(1, 800, 800, 1)
    side_len = 40
    cell_num = test_xr2.shape[1] // side_len

    cell_id = np.arange(1, cell_num ** 2 + 1)
    cell_id = np.random.choice(cell_id, cell_num ** 2, replace=False)
    cell_idx = 0
    for row in range(cell_num):
        for col in range(cell_num):
            test_xr2[0, row * side_len:(row + 1) * side_len, col * side_len:(col + 1) * side_len, :] = cell_id[cell_idx]
            cell_idx += 1

    ####
    row_num, col_num, crop_size, overlap_frac = test_xr2.shape[1], test_xr2.shape[2], 100, 0.2

    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=row_num, crop_size=crop_size,
                                                                   overlap_frac=overlap_frac)

    # generate a vector of random offsets to jitter the crop window, simulating mismatches between frames
    offset_len = 5
    row_offset = np.append(np.append(0, np.random.randint(-offset_len, offset_len, len(starts) - 2)), 0)
    col_offset = np.append(np.append(0, np.random.randint(-offset_len, offset_len, len(starts) - 2)), 0)

    # modify indices by random offset
    row_starts, row_ends = starts + row_offset, ends + row_offset
    col_starts, col_ends = starts + col_offset, ends + col_offset

    cropped, padded = npz_preprocessing.crop_helper(input_data=test_xr2, row_start=row_starts, row_end=row_ends,
                                                    col_start=col_starts, col_end=col_ends,
                                                    padding=((0, 0), (0, padding), (0, padding), (0, 0)))
    cropped_labels = cropped[..., -1:].values

    stitched_img = npz_postprocessing.stitch_crops(stack=cropped_labels, padded_img_shape=padded, row_starts=starts,
                                                   row_ends=ends, col_starts=starts, col_ends=ends)

    # trim padding
    stitched_img = stitched_img[:, :-padding, :-padding, :]

    relabeled = skimage.measure.label(stitched_img[0, :, :, 0])

    props = skimage.measure.regionprops_table(relabeled, properties=["area", "label"])

    # same number of unique objects before and after
    assert(len(np.unique(relabeled)) == len(np.unique(test_xr2[0, :, :, 0])))

    # no cell is smaller than offset subtracted from each side
    min_size = (side_len - offset_len * 2) ** 2
    max_size = (side_len + offset_len * 2) ** 2

    assert(np.all(props["area"] <= max_size))
    assert(np.all(props["area"] >= min_size))


# integration test for whole crop + stitch workflow pipeline
def test_crop_and_stitch():

    # create a test image with tiled unique values across the image
    test_xr = _blank_xr(2, 400, 400, 4)

    cell_idx = 1
    for i in range(12):
        for j in range(11):
            for img in range(test_xr.shape[0]):
                test_xr[img, (i * 35):(i * 35 + 10 + img * 10), (j * 37):(j * 37 + 8 + img * 10), 3] = cell_idx
            cell_idx += 1

    base_dir = "tests/caliban_toolbox/"
    test_xr.to_netcdf(os.path.join(base_dir, "test_xr.xr"))

    # crop data
    npz_preprocessing.crop_multichannel_data(xarray_path=os.path.join(base_dir, "test_xr.xr"),
                                             folder_save=os.path.join(base_dir, "test_folder"),
                                             crop_size=(150, 300), overlap_frac=0.2, relabel=True,
                                             blank_labels="separate")

    # stitch data
    npz_postprocessing.reconstruct_image_stack(crop_dir=base_dir + "test_folder")
    stitched_xr = xr.open_dataarray(os.path.join(base_dir, "test_folder", "stitched_images.nc"))

    # all the same pixels are marked
    assert(np.all(np.equal(stitched_xr[:, :, 0] > 0, test_xr[:, :, 0] > 0)))

    # there are the same number of cells
    assert(len(np.unique(stitched_xr)) == len(np.unique(test_xr)))

    # clean up
    shutil.rmtree(base_dir + "test_folder")
    os.remove(base_dir + "test_xr.xr")


# make sure correct indices are returned
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


# make sure input xarray is subset correctly
def test_montage_helper():

    # test output shape with even division of montage
    fov_len, montage_stack_len, crop_num, montage_num, row_len, col_len, chan_len = 1, 4, 1, 10, 50, 50, 3
    stack_len = 40

    montage_start_indices, montage_end_indices = npz_preprocessing.compute_montage_indices(stack_len, montage_stack_len, 0)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=montage_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    montage_output = npz_preprocessing.montage_helper(input_data, montage_start_indices, montage_end_indices)

    assert montage_output.shape == (fov_len, montage_stack_len, crop_num, int(np.ceil(stack_len / montage_stack_len)),
                                    row_len, col_len, chan_len)

    # test output shape with uneven division of montage
    fov_len, montage_stack_len, crop_num, montage_num, row_len, col_len, chan_len = 1, 6, 1, 10, 50, 50, 3
    stack_len = 40

    montage_start_indices, montage_end_indices = npz_preprocessing.compute_montage_indices(stack_len, montage_stack_len, 0)

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=montage_num,
                                 row_len=row_len, col_len=col_len, chan_len=chan_len)

    montage_output = npz_preprocessing.montage_helper(input_data, montage_start_indices, montage_end_indices)

    assert montage_output.shape == (fov_len, montage_stack_len, crop_num, (np.ceil(stack_len / montage_stack_len)),
                                    row_len, col_len, chan_len)

    # test output shape with montage overlaps
    fov_len, montage_stack_len, crop_num, montage_num, row_len, col_len, chan_len = 1, 6, 1, 10, 50, 50, 3
    stack_len = 40
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
    fov_len, montage_stack_len, crop_num, montage_num, row_len, col_len, chan_len = 1, 4, 1, 10, 50, 50, 3
    stack_len = 40

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


# test overall calling function
def test_create_montage_data():

    # test output shape with even division of montage
    fov_len, montage_stack_len, crop_num, montage_num, row_len, col_len, chan_len = 1, 4, 1, 10, 50, 50, 3
    stack_len = 40

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=montage_num,
                                 row_len=row_len, col_len=col_len, chan_len=chan_len)

    montage_xr, montage_indices = npz_preprocessing.create_montage_data(input_data, montage_stack_len)

    assert montage_xr.shape == (fov_len, montage_stack_len, crop_num, int(np.ceil(stack_len / montage_stack_len)),
                                row_len, col_len, chan_len)


def test_save_npzs_for_caliban():
    fov_len, montage_stack_len, crop_num, montage_num, row_len, col_len, chan_len = 1, 4, 1, 10, 50, 50, 3
    stack_len = 40

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=montage_stack_len,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    montage_xr, log_data = npz_preprocessing.create_montage_data(input_data, montage_stack_len)

    save_dir = "tests/caliban_toolbox/test_save_npzs_for_caliban/"
    npz_preprocessing.save_npzs_for_caliban(montage_xr, input_data, log_data, save_dir)

    # check that correct size was saved

    test_npz_labels = np.load(save_dir + "fov1_row_0_col_0_montage_0.npz")

    assert test_npz_labels["y"].shape == (fov_len, montage_stack_len, row_len, col_len, 1)

    assert test_npz_labels["y"].shape[:-1] == test_npz_labels["X"].shape[:-1]

    # check that json saved succesfully
    with open(os.path.join(save_dir, "log_data.json")) as json_file:
        saved_log_data = json.load(json_file)

    assert saved_log_data["original_shape"] == list(input_data.shape)
    shutil.rmtree(save_dir)


def test_load_montages():
    fov_len, montage_stack_len, montage_num, row_len, col_len, chan_len = 1, 4, 10, 50, 50, 3
    stack_len = 40

    input_data = _blank_stack_xr(fov_len=fov_len, stack_len=stack_len, row_len=row_len,
                                 col_len=col_len, chan_len=chan_len)

    montage_xr, montage_indices = npz_preprocessing.create_montage_data(input_data, montage_stack_len)

    # tag the upper left hand corner of the label in each montage
    tags = np.arange(int(np.ceil(stack_len / montage_stack_len)))
    montage_xr[0, 0, :, 0, 0, 2] = tags
    save_dir = "tests/caliban_toolbox/test_load_montages/"
    npz_preprocessing.save_npzs_for_caliban(montage_xr, montage_indices, save_dir)

    with open(save_dir + "montage_log_data.json") as json_file:
        montage_log_data = json.load(json_file)

    loaded_montages = npz_postprocessing.load_montages(save_dir, montage_log_data)

    assert np.all(np.equal(loaded_montages[0, 0, :, 0, 0, 0], tags))

    shutil.rmtree(save_dir)


def test_stitch_montages():

    # test case with even division
    fov_len, montage_stack_len, montage_num, row_len, col_len, chan_len = 1, 4, 10, 50, 50, 3
    stack_len = 40

    input_data = _blank_stack_xr(fov_len=fov_len, stack_len=stack_len, row_len=row_len,
                                 col_len=col_len, chan_len=chan_len)

    # tag upper left hand corner of the label in each image
    tags = np.arange(stack_len)
    input_data[0, :, 0, 0, 2] = tags

    montage_xr, montage_indices = npz_preprocessing.create_montage_data(input_data, montage_stack_len)
    fov_names = input_data.fovs.values

    montage_log_data = {}
    montage_log_data["montage_indices"] = montage_indices.tolist()
    montage_log_data["fov_names"] = fov_names.tolist()
    montage_log_data["montage_shape"] = montage_xr.shape
    montage_log_data["montage_num"] = montage_num

    stitched_montages = npz_postprocessing.stitch_montages(montage_xr[..., -1:], montage_log_data)

    assert np.all(np.equal(stitched_montages[0, :, 0, 0, 0], tags))


    # test case without even division
    fov_len, montage_stack_len, montage_num, row_len, col_len, chan_len = 1, 6, 10, 50, 50, 3
    stack_len = 40

    input_data = _blank_stack_xr(fov_len=fov_len, stack_len=stack_len, row_len=row_len,
                                 col_len=col_len, chan_len=chan_len)

    # tag upper left hand corner of the label in each image
    tags = np.arange(stack_len)
    input_data[0, :, 0, 0, 2] = tags

    montage_xr, montage_indices = npz_preprocessing.create_montage_data(input_data, montage_stack_len)
    fov_names = input_data.fovs.values

    montage_log_data = {}
    montage_log_data["montage_indices"] = montage_indices.tolist()
    montage_log_data["fov_names"] = fov_names.tolist()
    montage_log_data["montage_shape"] = montage_xr.shape
    montage_log_data["montage_num"] = montage_num

    stitched_montages = npz_postprocessing.stitch_montages(montage_xr[..., -1:], montage_log_data)

    assert np.all(np.equal(stitched_montages[0, :, 0, 0, 0], tags))


def test_reconstruct_montage_data():
    fov_len, montage_stack_len, montage_num, row_len, col_len, chan_len = 1, 4, 10, 50, 50, 3
    stack_len = 40

    input_data = _blank_stack_xr(fov_len=fov_len, stack_len=stack_len, row_len=row_len,
                                 col_len=col_len, chan_len=chan_len)
    # tag upper left hand corner of the label in each image
    tags = np.arange(stack_len)
    input_data[0, :, 0, 0, 2] = tags

    montage_xr, montage_indices = npz_preprocessing.create_montage_data(input_data, montage_stack_len)

    save_dir = "tests/caliban_toolbox/test_reconstruct_montage_data/"
    npz_preprocessing.save_npzs_for_caliban(montage_xr, montage_indices, save_dir)

    stitched_montages = npz_postprocessing.reconstruct_montage_data(save_dir)
    assert np.all(np.equal(stitched_montages[0, :, 0, 0, 0], tags))

    shutil.rmtree(save_dir)

