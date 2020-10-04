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

from caliban_toolbox.dataset_splitter import DatasetSplitter


def test__init__():
    seed = 123,
    splits = [0.5, 0.75, 1]

    ds = DatasetSplitter(seed=seed, splits=splits)

    assert ds.seed == seed
    assert ds.splits == splits

    # unsorted splits get sorted
    splits = [0.8, 0.3, 0.5]
    ds = DatasetSplitter(seed=seed, splits=splits)

    assert ds.splits == sorted(ds.splits)

    with pytest.raises(ValueError):
        # first split is size 0
        splits = [0, 0.25, 0.5]
        _ = DatasetSplitter(splits=splits)

    with pytest.raises(ValueError):
        # last split is greater than 1
        splits = [0.1, 0.25, 1.5]
        _ = DatasetSplitter(splits=splits)

    with pytest.raises(ValueError):
        # duplicate splits
        splits = [0.1, 0.1, 1]
        _ = DatasetSplitter(splits=splits)


def test__validate_dict():
    valid_dict = {'X': 1, 'y': 2}
    ds = DatasetSplitter()
    ds._validate_dict(valid_dict)

    invalid_dict = {'X': 1, 'y1': 2}
    with pytest.raises(ValueError):
        ds._validate_dict(invalid_dict)

    invalid_dict = {'X1': 1, 'y': 2}
    with pytest.raises(ValueError):
        ds._validate_dict(invalid_dict)


def test__duplicate_indices():
    test_indices = [np.arange(5), np.arange(1), np.arange(7)]
    min_size = 8

    for test_idx in test_indices:
        ds = DatasetSplitter()
        duplicated_indices = ds._duplicate_indices(indices=test_idx, min_size=min_size)

        assert len(duplicated_indices) == min_size
        # all of the same indices are still present
        assert set(test_idx) == set(duplicated_indices)


def test_split():
    X_vals = np.arange(100)
    y_vals = np.arange(100, 200)

    data_dict = {'X': X_vals, 'y': y_vals}

    splits = [0.1, 0.5, 1]
    ds = DatasetSplitter(splits=splits, seed=0)
    split_dict = ds.split(train_dict=data_dict)

    split_x_vals, split_y_vals = [], []
    for split in splits:
        current_split = split_dict[str(split)]

        assert len(current_split['X']) == int(100 * split)

        if split_x_vals == []:
            # first split
            split_x_vals = current_split['X']
            split_y_vals = current_split['y']
        else:
            # make sure all all previous values are in current split
            current_x_vals = current_split['X']
            current_y_vals = current_split['y']

            assert np.all(np.isin(split_x_vals, current_x_vals))
            assert np.all(np.isin(split_y_vals, current_y_vals))

            # update counter with current values
            split_x_vals = current_x_vals
            split_y_vals = current_y_vals

    # same seed should produce same values
    ds = DatasetSplitter(splits=splits, seed=0)
    split_dict_same_seed = ds.split(train_dict=data_dict)
    for split in split_dict_same_seed:
        current_split = split_dict_same_seed[split]
        original_split = split_dict[split]

        for data in current_split:
            assert np.array_equal(current_split[data], original_split[data])

    # differet seed should produce different values
    ds = DatasetSplitter(splits=splits, seed=1)
    split_dict_same_seed = ds.split(train_dict=data_dict)
    for split in split_dict_same_seed:
        current_split = split_dict_same_seed[split]
        original_split = split_dict[split]

        for data in current_split:
            assert not np.array_equal(current_split[data], original_split[data])

    # split corresponding to fewer than 1 image returns a single image
    splits = [0.001, 0.3, 1]
    ds = DatasetSplitter(splits=splits, seed=0)
    split_dict = ds.split(train_dict=data_dict)
    print(split_dict['0.001']['X'])
    assert len(split_dict['0.001']['X']) == 1

    # setting minimum size
    split_dict = ds.split(train_dict=data_dict, min_size=10)
    assert len(split_dict['0.001']['X']) == 10
