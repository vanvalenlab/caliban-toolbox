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


def test_load_npzs():
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 4

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    slice_xr, log_data = npz_preprocessing.create_slice_data(input_data, slice_stack_len)

    crop_size = (10, 10)
    overlap_frac = 0.2
    data_xr_cropped, log_data_crop = npz_preprocessing.crop_multichannel_data(data_xr=slice_xr, crop_size=crop_size,
                                                                              overlap_frac=overlap_frac,
                                                                              test_parameters=False)
    combined_log_data = {**log_data, **log_data_crop}
    # tag the upper left hand corner of the label in each slice
    slice_tags = np.arange(data_xr_cropped.shape[3])
    crop_tags = np.arange(data_xr_cropped.shape[2])
    data_xr_cropped[0, 0, :, 0, 0, 0, 2] = crop_tags
    data_xr_cropped[0, 0, 0, :, 0, 0, 2] = slice_tags
    save_dir = "tests/caliban_toolbox/test_load_slices/"

    npz_preprocessing.save_npzs_for_caliban(resized_xr=data_xr_cropped, original_xr=input_data,
                                            log_data=combined_log_data, save_dir=save_dir,
                                            blank_labels="include", save_format="npz")

    with open(os.path.join(save_dir, "log_data.json")) as json_file:
        saved_log_data = json.load(json_file)

    loaded_slices = npz_postprocessing.load_npzs(save_dir, saved_log_data)

    assert np.all(np.equal(loaded_slices[0, 0, :, 0, 0, 0, 0], crop_tags))
    assert np.all(np.equal(loaded_slices[0, 0, 0, :, 0, 0, 0], slice_tags))

    shutil.rmtree(save_dir)

    # test slices with unequal last length
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 7

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    slice_xr, log_data = npz_preprocessing.create_slice_data(input_data, slice_stack_len)

    crop_size = (10, 10)
    overlap_frac = 0.2
    data_xr_cropped, log_data_crop = npz_preprocessing.crop_multichannel_data(data_xr=slice_xr, crop_size=crop_size,
                                                                              overlap_frac=overlap_frac,
                                                                              test_parameters=False)
    # tag the upper left hand corner of the label in each slice
    slice_tags = np.arange(data_xr_cropped.shape[3])
    crop_tags = np.arange(data_xr_cropped.shape[2])
    data_xr_cropped[0, 0, :, 0, 0, 0, 2] = crop_tags
    data_xr_cropped[0, 0, 0, :, 0, 0, 2] = slice_tags
    save_dir = "tests/caliban_toolbox/test_load_slices/"

    combined_log_data = {**log_data, **log_data_crop}

    npz_preprocessing.save_npzs_for_caliban(resized_xr=data_xr_cropped, original_xr=input_data,
                                            log_data=combined_log_data, save_dir=save_dir,
                                            blank_labels="include", save_format="npz")

    loaded_slices = npz_postprocessing.load_npzs(save_dir, combined_log_data)

    assert np.all(np.equal(loaded_slices[0, 0, :, 0, 0, 0, 0], crop_tags))
    assert np.all(np.equal(loaded_slices[0, 0, 0, :, 0, 0, 0], slice_tags))

    shutil.rmtree(save_dir)


def test_stitch_crops():
    # generate stack of crops from image with grid pattern
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 2, 1, 1, 1, 400, 400, 4

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    # create image with
    cell_idx = 1
    for i in range(12):
        for j in range(11):
            for fov in range(input_data.shape[0]):
                input_data[fov, :, :, :, (i * 35):(i * 35 + 10 + fov * 10), (j * 37):(j * 37 + 8 + fov * 10), 3] = cell_idx
            cell_idx += 1

    crop_size, overlap_frac = 50, 0.2

    cropped, log_data = npz_preprocessing.crop_multichannel_data(data_xr=input_data, crop_size=(crop_size, crop_size),
                                                    overlap_frac=overlap_frac)
    cropped_labels = cropped[..., -1:].values
    log_data["original_shape"] = input_data.shape

    stitched_img = npz_postprocessing.stitch_crops(annotated_data=cropped_labels, log_data=log_data)

    # trim padding
    row_padding, col_padding = log_data["row_padding"], log_data["col_padding"]
    stitched_img = stitched_img[:, :, :, :, :-row_padding, :-col_padding, :]

    # check that objects are at same location
    assert(np.all(np.equal(stitched_img[..., 0] > 0, input_data.values[..., 3] > 0)))

    # check that same number of unique objects
    assert len(np.unique(stitched_img)) == len(np.unique(input_data.values))

    # test stitching imperfect annotator labels that slightly overlap
    # generate stack of crops from image with grid pattern
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 1, 1, 1, 800, 800, 1

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)
    side_len = 40
    cell_num = input_data.shape[4] // side_len

    cell_id = np.arange(1, cell_num ** 2 + 1)
    cell_id = np.random.choice(cell_id, cell_num ** 2, replace=False)
    cell_idx = 0
    for row in range(cell_num):
        for col in range(cell_num):
            input_data[0, 0, 0, 0, row * side_len:(row + 1) * side_len, col * side_len:(col + 1) * side_len, 0] = cell_id[cell_idx]
            cell_idx += 1

    crop_size, overlap_frac = 100, 0.2

    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=row_len, crop_size=crop_size,
                                                                   overlap_frac=overlap_frac)

    # generate a vector of random offsets to jitter the crop window, simulating mismatches between frames
    offset_len = 5
    row_offset = np.append(np.append(0, np.random.randint(-offset_len, offset_len, len(starts) - 2)), 0)
    col_offset = np.append(np.append(0, np.random.randint(-offset_len, offset_len, len(starts) - 2)), 0)

    # modify indices by random offset
    row_starts, row_ends = starts + row_offset, ends + row_offset
    col_starts, col_ends = starts + col_offset, ends + col_offset

    cropped, padded = npz_preprocessing.crop_helper(input_data=input_data, row_starts=row_starts, row_ends=row_ends,
                                                    col_starts=col_starts, col_ends=col_ends,
                                                    padding=(padding, padding))

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

    stitched_img = npz_postprocessing.stitch_crops(annotated_data=cropped_labels, log_data=log_data)

    # trim padding
    stitched_img = stitched_img[:, :, :, :, :-padding, :-padding, :]

    relabeled = skimage.measure.label(stitched_img[0, 0, 0, 0, :, :, 0])

    props = skimage.measure.regionprops_table(relabeled, properties=["area", "label"])

    # same number of unique objects before and after
    assert(len(np.unique(relabeled)) == len(np.unique(input_data[0, 0, 0, 0, :, :, 0])))

    # no cell is smaller than offset subtracted from each side
    min_size = (side_len - offset_len * 2) ** 2
    max_size = (side_len + offset_len * 2) ** 2

    assert(np.all(props["area"] <= max_size))
    assert(np.all(props["area"] >= min_size))


def test_reconstruct_image_data():
    # generate stack of crops from image with grid pattern
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 2, 1, 1, 1, 400, 400, 4

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    # create image with
    cell_idx = 1
    for i in range(12):
        for j in range(11):
            for fov in range(input_data.shape[0]):
                input_data[fov, :, :, :, (i * 35):(i * 35 + 10 + fov * 10), (j * 37):(j * 37 + 8 + fov * 10),
                3] = cell_idx
            cell_idx += 1

    crop_size, overlap_frac = 50, 0.2
    save_dir = "tests/caliban_toolbox/test_crop_and_stitch"

    # crop data
    data_xr_cropped, log_data = npz_preprocessing.crop_multichannel_data(data_xr=input_data,
                                                                         crop_size=(crop_size, crop_size),
                                                                         overlap_frac=0.2)

    # stitch data
    npz_preprocessing.save_npzs_for_caliban(resized_xr=data_xr_cropped, original_xr=input_data, log_data=log_data,
                                            save_dir=save_dir)

    npz_postprocessing.reconstruct_image_stack(crop_dir=save_dir)

    stitched_xr = xr.open_dataarray(os.path.join(save_dir, "stitched_images.nc"))

    # all the same pixels are marked
    assert(np.all(np.equal(stitched_xr[:, :, 0] > 0, input_data[:, :, 0] > 0)))

    # there are the same number of cells
    assert(len(np.unique(stitched_xr)) == len(np.unique(input_data)))

    # clean up
    shutil.rmtree(save_dir)


def test_stitch_slices():
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 4

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    # generate ordered data
    linear_seq = np.arange(stack_len * row_len * col_len)
    test_vals = linear_seq.reshape((stack_len, row_len, col_len))
    input_data[0, :, 0, 0, :, :, 2] = test_vals

    slice_xr, log_data = npz_preprocessing.create_slice_data(input_data, slice_stack_len)

    # TODO move crop + slice testing to another test function
    crop_size = (10, 10)
    overlap_frac = 0.2
    data_xr_cropped, log_data_crop = npz_preprocessing.crop_multichannel_data(data_xr=slice_xr, crop_size=crop_size,
                                                                              overlap_frac=overlap_frac,
                                                                              test_parameters=False)

    # # get parameters
    # row_crop_size, col_crop_size = crop_size[0], crop_size[1]
    # num_row_crops, num_col_crops = log_data_crop["num_row_crops"], log_data_crop["num_col_crops"]
    # num_slices = log_data["num_slices"]
    log_data["original_shape"] = input_data.shape
    log_data["fov_names"] = input_data.fovs.values
    stitched_slices = npz_postprocessing.stitch_slices(slice_xr[..., -1:], {**log_data})

    assert np.all(np.equal(stitched_slices[0, :, 0, 0, :, :, 0], test_vals))


    # test case without even division
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 7

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    # generate ordered data
    linear_seq = np.arange(stack_len * row_len * col_len)
    test_vals = linear_seq.reshape((stack_len, row_len, col_len))
    input_data[0, :, 0, 0, :, :, 2] = test_vals

    slice_xr, log_data = npz_preprocessing.create_slice_data(input_data, slice_stack_len)

    # get parameters
    log_data["original_shape"] = input_data.shape
    log_data["fov_names"] = input_data.fovs.values
    stitched_slices = npz_postprocessing.stitch_slices(slice_xr[..., -1:], log_data)

    assert np.all(np.equal(stitched_slices[0, :, 0, 0, :, :, 0], test_vals))


def test_reconstruct_slice_data():
    fov_len, stack_len, crop_num, slice_num, row_len, col_len, chan_len = 1, 40, 1, 1, 50, 50, 3
    slice_stack_len = 4

    input_data = _blank_data_xr(fov_len=fov_len, stack_len=stack_len, crop_num=crop_num, slice_num=slice_num,
                                row_len=row_len, col_len=col_len, chan_len=chan_len)

    # tag upper left hand corner of the label in each image
    tags = np.arange(stack_len)
    input_data[0, :, 0, 0, 0, 0, 2] = tags

    slice_xr, slice_log_data = npz_preprocessing.create_slice_data(input_data, slice_stack_len)

    save_dir = "tests/caliban_toolbox/test_reconstruct_slice_data/"
    npz_preprocessing.save_npzs_for_caliban(resized_xr=slice_xr, original_xr=input_data,
                                            log_data={**slice_log_data}, save_dir=save_dir, blank_labels="include",
                                            save_format="npz")

    stitched_slices = npz_postprocessing.reconstruct_slice_data(save_dir)
    assert np.all(np.equal(stitched_slices[0, :, 0, 0, 0, 0, 0], tags))

    shutil.rmtree(save_dir)
