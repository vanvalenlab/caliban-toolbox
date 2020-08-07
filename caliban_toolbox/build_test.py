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
import pytest

import numpy as np

from caliban_toolbox import build


def _make_npzs(sizes, num_images):
    npz_list = []

    for i in range(len(num_images)):
        x = np.zeros((num_images[i], ) + sizes[i] + (4, ))
        y = np.zeros((num_images[i], ) + sizes[i] + (1, ))
        npz = {'X': x, 'y': y}

        npz_list.append(npz)

    return npz_list


def test_compute_cell_size():
    labels = np.zeros((3, 40, 40, 1), dtype='int')
    labels[0, :10, :10, 0] = 1
    labels[0, 20:25, 20:25, 0] = 2
    labels[0, 12:14, 16:18] = 3

    labels[1, :4, :4, 0] = 2
    labels[1, 30:35, 35:40, 0] = 3
    labels[1, 36:39, 30:33, 0] = 1

    labels[2, 30:39, 20:29] = 1

    example_npz = {'y': labels}

    cell_sizes = build.compute_cell_size(npz_file=example_npz, method='median', by_image=True)
    assert np.all(cell_sizes == [25, 16, 81])

    cell_sizes = build.compute_cell_size(npz_file=example_npz, method='median', by_image=False)
    assert cell_sizes == [25]

    cell_sizes = build.compute_cell_size(npz_file=example_npz, method='mean', by_image=True)
    assert np.all(np.round(cell_sizes, 2) == [43, 16.67, 81])

    cell_sizes = build.compute_cell_size(npz_file=example_npz, method='mean', by_image=False)
    assert np.round(cell_sizes, 2) == [37.14]

    # incorrect method
    with pytest.raises(ValueError):
        _ = build.compute_cell_size(npz_file=example_npz, method='bad_method', by_image=True)

    # incorrect input data
    with pytest.raises(ValueError):
        _ = build.compute_cell_size(npz_file={'y': labels[0]}, method='bad_method', by_image=True)


def test_reshape_training_image():
    # test without resizing or cropping
    X_data, y_data = np.zeros((5, 40, 40, 3)), np.zeros((5, 40, 40, 2))
    resize_ratio = 1
    final_size = (40, 40)
    stride_ratio = 1

    reshaped_X, reshaped_y = build.reshape_training_image(X_data=X_data,
                                                          y_data=y_data,
                                                          resize_ratio=resize_ratio,
                                                          final_size=final_size,
                                                          stride_ratio=stride_ratio)
    assert reshaped_X.shape == X_data.shape
    assert reshaped_y.shape == y_data.shape

    # test with just cropping
    X_data, y_data = np.zeros((5, 80, 40, 3)), np.zeros((5, 80, 40, 2))
    resize_ratio = 1
    final_size = (40, 40)
    stride_ratio = 1

    reshaped_X, reshaped_y = build.reshape_training_image(X_data=X_data,
                                                          y_data=y_data,
                                                          resize_ratio=resize_ratio,
                                                          final_size=final_size,
                                                          stride_ratio=stride_ratio)
    assert list(reshaped_X.shape) == [X_data.shape[0] * 2] + list(final_size) + [X_data.shape[-1]]
    assert list(reshaped_y.shape) == [y_data.shape[0] * 2] + list(final_size) + [y_data.shape[-1]]

    # test with just resizing
    X_data, y_data = np.zeros((5, 40, 40, 3)), np.zeros((5, 40, 40, 2))
    resize_ratio = 1
    final_size = (80, 80)
    stride_ratio = 2

    reshaped_X, reshaped_y = build.reshape_training_image(X_data=X_data,
                                                          y_data=y_data,
                                                          resize_ratio=resize_ratio,
                                                          final_size=final_size,
                                                          stride_ratio=stride_ratio)
    assert list(reshaped_X.shape) == [X_data.shape[0]] + list(final_size) + [X_data.shape[-1]]
    assert list(reshaped_y.shape) == [y_data.shape[0]] + list(final_size) + [y_data.shape[-1]]

    # test with resizing and cropping
    X_data, y_data = np.zeros((5, 40, 40, 3)), np.zeros((5, 40, 40, 2))
    resize_ratio = 2
    final_size = (40, 40)
    stride_ratio = 2

    reshaped_X, reshaped_y = build.reshape_training_image(X_data=X_data,
                                                          y_data=y_data,
                                                          resize_ratio=resize_ratio,
                                                          final_size=final_size,
                                                          stride_ratio=stride_ratio)
    assert list(reshaped_X.shape) == [X_data.shape[0] * 4] + list(X_data.shape[1:])
    assert list(reshaped_y.shape) == [y_data.shape[0] * 4] + list(y_data.shape[1:])


def test_pad_image_stack():
    # rows and cols both need to be modified
    input_stack = np.zeros((2, 55, 55, 2))
    tags = [1, 2]
    input_stack[:, 0, 0, 0] = tags
    crop_size = (10, 10)
    padded_stack = build.pad_image_stack(images=input_stack, crop_size=crop_size)
    assert padded_stack.shape == (2, 60, 60, 2)
    assert np.all(padded_stack[:, 0, 0, 0] == tags)

    # just rows need to be modified
    input_stack = np.zeros((2, 50, 35, 2))
    input_stack[:, 0, 0, 0] = tags
    crop_size = (10, 10)
    padded_stack = build.pad_image_stack(images=input_stack, crop_size=crop_size)
    assert padded_stack.shape == (2, 50, 40, 2)
    assert np.all(padded_stack[:, 0, 0, 0] == tags)

    # neither needs to be modified
    input_stack = np.zeros((2, 30, 50, 2))
    input_stack[:, 0, 0, 0] = tags
    crop_size = (10, 10)
    padded_stack = build.pad_image_stack(images=input_stack, crop_size=crop_size)
    assert padded_stack.shape == input_stack.shape
    assert np.all(padded_stack[:, 0, 0, 0] == tags)


def test_combine_npz_files():
    # NPZ files are appropriate size and resolution
    num_images = [2, 2]
    sizes = [(256, 256), (256, 256)]
    npz_list = _make_npzs(sizes=sizes, num_images=num_images)
    resize_ratios = [[1], [1]]
    final_size = (256, 256)

    combined_x, combined_y = build.combine_npz_files(npz_list=npz_list,
                                                     resize_ratios=resize_ratios,
                                                     final_size=final_size)

    # check that correct number of NPZs present
    assert combined_x.shape[0] == np.sum(num_images)

    # check correct size of NPZs
    assert combined_x.shape[1:3] == final_size

    # NPZ files need to be cropped
    num_images = [2, 2]
    sizes = [(512, 512), (512, 512)]
    npz_crop_list = _make_npzs(sizes=sizes, num_images=num_images)
    resize_ratios = [[1], [1]]
    final_size = (256, 256)

    combined_x, combined_y = build.combine_npz_files(npz_list=npz_crop_list,
                                                     resize_ratios=resize_ratios,
                                                     final_size=final_size)

    # check that correct number of NPZs present
    assert combined_x.shape[0] == np.sum(num_images) * 4

    # check correct size of NPZs
    assert combined_x.shape[1:3] == final_size

    # NPZ files need to be resized
    num_images = [2, 2]
    sizes = [(128, 128), (128, 128)]
    npz_resize_list = _make_npzs(sizes=sizes, num_images=num_images)
    resize_ratios = [[2], [2]]
    final_size = (256, 256)

    combined_x, combined_y = build.combine_npz_files(npz_list=npz_resize_list,
                                                     resize_ratios=resize_ratios,
                                                     final_size=final_size)

    # check that correct number of NPZs present
    assert combined_x.shape[0] == np.sum(num_images)

    # check correct size of NPZs
    assert combined_x.shape[1:3] == final_size

    # some need to be cropped, some need to be resized
    npz_list = npz_crop_list + npz_resize_list
    resize_ratios = [[1], [1], [2], [2]]
    final_size = (256, 256)

    combined_npz = build.combine_npz_files(npz_list=npz_list, resize_ratios=resize_ratios,
                                           final_size=final_size)

    combined_x, combined_y = combined_npz

    # check that correct number of NPZs present
    assert combined_x.shape[0] == (np.sum(num_images) + np.sum(num_images) * 4)

    # check correct size of NPZs
    assert combined_x.shape[1:3] == final_size

    # different resizing for each image in the NPZ
    num_images = [2, 2]
    sizes = [(256, 256), (256, 256)]
    npz_resize_list = _make_npzs(sizes=sizes, num_images=num_images)
    resize_ratios = [[1], [1, 2]]
    final_size = (256, 256)

    combined_x, combined_y = build.combine_npz_files(npz_list=npz_resize_list,
                                                     resize_ratios=resize_ratios,
                                                     final_size=final_size)

    # check that correct number of NPZs present
    assert combined_x.shape[0] == np.sum(2 + 1 + 4)

    # check correct size of NPZs
    assert combined_x.shape[1:3] == final_size
