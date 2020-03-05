import os
import shutil
import json

import numpy as np
from caliban_toolbox.pre_annotation import npz_preprocessing
from caliban_toolbox.post_annotation import npz_postprocessing
import xarray as xr

import importlib
importlib.reload(npz_preprocessing)
importlib.reload(npz_postprocessing)


def _blank_xr(fov_num, row_num, col_num, chan_num):
    """Test function to generate a blank xarray with the supplied dimensions

    Inputs
        fov_num: number of distinct FOVs
        row_num: number of rows
        col_num: number of cols
        chan_num: number of channels

    Outputs
        test_xr: xarray of [fov_num, row_num, col_num, chan_num]"""

    test_img = np.zeros((fov_num, row_num, col_num, chan_num))

    fovs = ["fov" + str(x) for x in range(1, fov_num + 1)]
    channels = ["channel" + str(x) for x in range(1, chan_num + 1)]

    test_xr = xr.DataArray(data=test_img, coords=[fovs, range(row_num), range(col_num), channels],
                           dims=["fovs", "rows", "cols", "channels"])

    return test_xr


def _blank_cropped_xr(fov_num, crop_num, row_len, col_len, chan_num):
    """Test function to generate a blank xarray with the supplied dimensions

    Inputs
        fov_num: number of distinct FOVs
        crop_num: number of distinct crops
        row_len: number of rows
        col_len: number of cols
        chan_num: number of channels

    Outputs
        test_xr: xarray of [fov_num, row_num, col_num, chan_num]"""

    test_img = np.zeros((fov_num, crop_num, row_len, col_len, chan_num))

    fovs = ["fov" + str(x) for x in range(1, fov_num + 1)]
    channels = ["channel" + str(x) for x in range(1, chan_num + 1)]

    test_xr = xr.DataArray(data=test_img, coords=[fovs, range(crop_num), range(row_len), range(col_len), channels],
                           dims=["fovs", "crops", "rows", "cols", "channels"])

    return test_xr


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
    fov_num, row_num, col_num, chan_num = 2, 200, 200, 1
    crop_size, overlap_frac = 200, 0.2

    # test only one crop
    test_xr = _blank_xr(fov_num=fov_num, row_num=row_num, col_num=col_num, chan_num=chan_num)

    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=row_num, crop_size=crop_size,
                                                                               overlap_frac=overlap_frac)
    cropped, padded = npz_preprocessing.crop_helper(input_data=test_xr, row_start=starts, row_end=ends,
                                                    col_start=starts, col_end=ends,
                                                    padding=((0, 0), (0, padding), (0, padding), (0, 0)))

    assert(cropped.shape == (fov_num, 1, row_num, col_num, chan_num))

    # test crops of different row/col dimensions
    row_crop, col_crop = 50, 40
    row_starts, row_ends, row_padding = npz_preprocessing.compute_crop_indices(img_len=row_num, crop_size=row_crop,
                                                                               overlap_frac=overlap_frac)

    col_starts, col_ends, col_padding = npz_preprocessing.compute_crop_indices(img_len=col_num, crop_size=col_crop,
                                                                               overlap_frac=overlap_frac)

    cropped, padded = npz_preprocessing.crop_helper(input_data=test_xr, row_start=row_starts, row_end=row_ends,
                                                    col_start=col_starts, col_end=col_ends,
                                                    padding=((0, 0), (0, row_padding), (0, col_padding), (0, 0)))

    assert(cropped.shape == (fov_num, 30, row_crop, col_crop, chan_num))

    # test that correct region of image is being cropped
    row_crop, col_crop = 40, 40

    # assign each pixel in the image a unique value
    linear_sequence = np.arange(0, fov_num * row_num * col_num * chan_num)
    linear_sequence_reshaped = np.reshape(linear_sequence, (fov_num, row_num, col_num, chan_num))
    test_xr[:, :, :, :] = linear_sequence_reshaped

    # crop the image
    row_starts, row_ends, row_padding = npz_preprocessing.compute_crop_indices(img_len=row_num, crop_size=row_crop,
                                                                               overlap_frac=overlap_frac)

    col_starts, col_ends, col_padding = npz_preprocessing.compute_crop_indices(img_len=col_num, crop_size=col_crop,
                                                                               overlap_frac=overlap_frac)

    cropped, padded = npz_preprocessing.crop_helper(input_data=test_xr, row_start=row_starts, row_end=row_ends,
                                                    col_start=col_starts, col_end=col_ends,
                                                    padding=((0, 0), (0, row_padding), (0, col_padding), (0, 0)))

    # check that the values of each crop match the value in uncropped image
    for img in range(test_xr.shape[0]):
        crop_counter = 0
        for row in range(len(row_starts)):
            for col in range(len(col_starts)):
                crop = cropped[img, crop_counter, :, :, 0].values

                original_image_crop = test_xr[img, row_ends[row]:row_ends[row], col_starts[col]:col_ends[col], :].values
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

    # create test_xr, give all crops constant values
    fov_num, row_num, col_num, chan_num = 2, 200, 200, 1
    test_xr = _blank_xr(fov_num=fov_num, row_num=row_num, col_num=col_num, chan_num=chan_num)
    test_xr[:, :, :, 0] = 1

    base_dir = "tests/caliban_toolbox/"
    test_xr.to_netcdf(os.path.join(base_dir, "crop_multichannel_test_xr.xr"))

    npz_preprocessing.crop_multichannel_data(xarray_path=os.path.join(base_dir, "crop_multichannel_test_xr.xr"),
                                             folder_save=os.path.join(base_dir, "crop_multichannel_test"),
                                             crop_size=(50, 50), overlap_frac=0.2, blank_labels="include",
                                             save_format="xr", relabel=True)

    # unpack JSON data
    with open(os.path.join(base_dir, "crop_multichannel_test", "log_data.json")) as json_file:
        log_data = json.load(json_file)

    assert(np.all(log_data["fov_names"] == test_xr.fovs.values))

    shutil.rmtree(os.path.join(base_dir, "crop_multichannel_test"))
    os.remove((os.path.join(base_dir, "crop_multichannel_test_xr.xr")))


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
    test_xr2 = _blank_xr(1, 400, 400, 1)
    side_len = 40
    cell_num = test_xr2.shape[1] // side_len

    cell_id = 1
    for row in range(cell_num):
        for col in range(cell_num):
            test_xr2[0, row*side_len:(row + 1)*side_len, col*side_len:(col + 1)*side_len, :] = cell_id
            cell_id += 1


    ####
    row_num, col_num, crop_size, overlap_frac = 400, 400, 100, 0.2

    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=row_num, crop_size=crop_size,
                                                                   overlap_frac=overlap_frac)

    # generate a vector of random offsets to jitter the crop window, simulating mismatches between frames
    row_offset = np.append(np.append(0, np.random.randint(-5, 5, len(starts) - 2)), 0)
    col_offset = np.append(np.append(0, np.random.randint(-5, 5, len(starts) - 2)), 0)

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
    io.imshow(stitched_img[0, :, :, 0])

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
