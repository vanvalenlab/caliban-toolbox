import os
import shutil

import numpy as np
from deepcell_toolbox.pre_annotation import npz_preprocessing
from deepcell_toolbox.post_annotation import npz_postprocessing
import importlib
importlib.reload(npz_preprocessing)


# tests for npz version of the pipeline
def test_compute_crop_indices():

    # test corner case of only one crop
    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=100, crop_size=100, overlap_frac=0.2)
    assert(len(starts) == 1)
    assert(len(ends) == 1)

    # test crop size that doesn't divide evenly into image size
    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=105, crop_size=20, overlap_frac=0.2)
    crop_num = np.ceil(105 / (20 - 20 * .2))
    assert(len(starts) == crop_num)
    assert(len(ends) == crop_num)
    assert(ends[-1] == crop_num * (20 - 20 * 0.2) + 20 * 0.2)

    # test overlap of 0 between crops
    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=200, crop_size=20, overlap_frac=0)
    assert (np.all(starts == range(0, 200, 20)))
    assert (np.all(ends == range(20, 201, 20)))
    assert (padding == 0)


def test_crop_images():

    # test only one crop
    test_img = np.zeros((200, 200, 1))
    starts, ends, padding = npz_preprocessing.compute_crop_indices(img_len=200, crop_size=200,
                                                                               overlap_frac=0.2)
    cropped, padded = npz_preprocessing.crop_images(input_data=test_img, row_start=starts, row_end=ends,
                                                    col_start=starts, col_end=ends,
                                                    padding=((0, padding), (0, padding), (0, 0)))

    assert(cropped.shape == (1, 200, 200, 1))

    # test crops of different row/col dimensions
    test_img = np.zeros((200, 200, 1))
    row_starts, row_ends, row_padding = npz_preprocessing.compute_crop_indices(img_len=200, crop_size=40,
                                                                               overlap_frac=0.2)

    col_starts, col_ends, col_padding = npz_preprocessing.compute_crop_indices(img_len=200, crop_size=50,
                                                                               overlap_frac=0.2)

    cropped, padded = npz_preprocessing.crop_images(input_data=test_img, row_start=row_starts, row_end=row_ends,
                                                    col_start=col_starts, col_end=col_ends,
                                                    padding=((0, row_padding), (0, col_padding), (0, 0)))

    assert(cropped.shape == (30, 40, 50, 1))

    # TODO: Test crop_npz function, check JSON outputs

    # TODO: test stitch_image function


# integration test for whole crop + stitch workflow pipeline
def test_crop_and_stitch():
    # create a test image with tiled unique values across the image
    test_img = np.zeros((400, 400, 1))
    cell_idx = 1
    for i in range(12):
        for j in range(11):
            test_img[(i * 35):(i * 35 + 10), (j * 37):(j * 37 + 8), 0] = cell_idx
            cell_idx += 1

    base_dir = "tests/deepcell_toolbox/"
    np.savez(base_dir + "test.npz", X=np.zeros((400, 400, 3)), y=test_img)

    npz_preprocessing.crop_npz(npz_name="test.npz", base_dir=base_dir, save_name="test_folder", crop_size=(200, 200),
                               overlap_frac=0.2)

    npz_postprocessing.reconstruct_npz(npz_dir=base_dir + "test_folder", original_npz=base_dir + "test.npz")

    final_npz = np.load(os.path.join(base_dir, "test_folder", "stitched_npz.npz"))
    stitched = final_npz["y"]

    # all the same pixels are marked
    assert(np.all(np.equal(stitched[:, :, 0] > 0, test_img[:, :, 0] > 0)))

    # there are the same number of cells
    assert(len(np.unique(stitched)) == len(np.unique(test_img)))

    # clean up
    os.remove(base_dir + "test_folder")
