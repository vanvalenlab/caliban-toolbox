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
import json
import pytest

import numpy as np


from caliban_toolbox.DatasetBuilder import train_val_test_split, DatasetBuilder


def _create_test_npz(path, constant_value=1, X_shape=(10, 20, 20, 3), y_shape=(10, 20, 20, 1)):
    X_data = np.full(X_shape, constant_value)
    y_data = np.full(y_shape, constant_value)
    np.savez(path, X=X_data, y=y_data)


def _create_test_dataset(path, experiments, tissues, platforms, npz_num):
    """Creates an example directory to load data from

    Args:
        path: folder to hold datasets
        experiments: list of experiment names
        tissues: list of tissue types for each experiment
        platforms: list of platform types for each experiment
        npz_num: number of unique NPZ files within each experiment. The NPZs within
            each experiment are constant-valued arrays corresponding to the index of that exp

    Raises:
        ValueError: If tissue_list, platform_list, or NPZ_num have different lengths
    """
    lengths = [len(x) for x in [experiments, tissues, platforms, npz_num]]
    if len(set(lengths)) != 1:
        raise ValueError('All inputs must have the same length')

    for i in range(len(experiments)):
        experiment_folder = os.path.join(path, experiments[i])
        os.makedirs(experiment_folder)

        metadata = dict()
        metadata['tissue'] = tissues[i]
        metadata['platform'] = platforms[i]

        metadata_path = os.path.join(experiment_folder, 'metadata.json')

        with open(metadata_path, 'w') as write_file:
            json.dump(metadata, write_file)

        for npz in range(npz_num[i]):
            _create_test_npz(path=os.path.join(experiment_folder, 'sub_exp_{}.npz'.format(npz)),
                             constant_value=i)


def _create_test_dict(tissues, platforms):
    data = []
    for i in range(len(tissues)):
        current_data = np.full((5, 40, 40, 3), i)
        data.append(current_data)

    data = np.concatenate(data, axis=0)
    X_data = data
    y_data = data[..., :1]

    tissue_list = [tissues[i] for i in range(len(tissues)) for _ in range(5)]
    platform_list = [platforms[i] for i in range(len(platforms)) for _ in range(5)]

    return {'X': X_data, 'y': y_data, 'tissue_list': tissue_list, 'platform_list': platform_list}


def mocked_compute_cell_size(data_dict):
    """Mocks compute cell size so we don't need to create synthetic data with correct cell size"""
    X = data_dict['X']
    constant_val = X[0, 0, 0, 0]

    multiplier = 400 + (400 * constant_val)

    return multiplier


def test__init__(tmp_path):
    # no folders in dataset
    with pytest.raises(ValueError):
        _ = DatasetBuilder(dataset_path=tmp_path)

    # single folder
    os.makedirs(os.path.join(tmp_path, 'example_folder'))
    db = DatasetBuilder(dataset_path=tmp_path)

    assert db.dataset_path == tmp_path

    # bad path
    with pytest.raises(ValueError):
        _ = DatasetBuilder(dataset_path='bad_path')


def test__identify_tissue_and_platform_types(tmp_path):
    # create dataset
    experiments = ['exp{}'.format(i) for i in range(5)]
    tissues = ['tissue1', 'tissue2', 'tissue3', 'tissue2', 'tissue1']
    platforms = ['platform1', 'platform1', 'platform2', 'platform2', 'platform3']
    npz_num = [1] * 5
    _create_test_dataset(tmp_path, experiments=experiments, tissues=tissues,
                         platforms=platforms, npz_num=npz_num)

    db = DatasetBuilder(dataset_path=tmp_path)

    db._identify_tissue_and_platform_types()

    # check that all tissues and platforms added
    assert set(db.all_tissues) == set(tissues)
    assert set(db.all_platforms) == set(platforms)


def test__load_experiment_single_npz(tmp_path):
    experiments, tissues, platforms, npz_num = ['exp1'], ['tissue1'], ['platform1'], [1]
    _create_test_dataset(tmp_path, experiments=experiments, tissues=tissues,
                         platforms=platforms, npz_num=npz_num)

    # initialize db
    db = DatasetBuilder(tmp_path)

    # load dataset
    X, y, tissue, platform = db._load_experiment(os.path.join(tmp_path, experiments[0]))

    # A single NPZ with 10 images
    assert X.shape[0] == 10
    assert y.shape[0] == 10

    assert tissue == tissues[0]
    assert platform == platforms[0]


def test__load_experiment_multiple_npz(tmp_path):
    experiments, tissues, platforms, npz_num = ['exp1'], ['tissue1'], ['platform1'], [5]
    _create_test_dataset(tmp_path, experiments=experiments, tissues=tissues,
                         platforms=platforms, npz_num=npz_num)

    # initialize db
    db = DatasetBuilder(tmp_path)

    # load dataset
    X, y, tissue, platform = db._load_experiment(os.path.join(tmp_path, experiments[0]))

    # 5 NPZs with 10 images each
    assert X.shape[0] == 50
    assert y.shape[0] == 50

    assert tissue == tissues[0]
    assert platform == platforms[0]


def test__load_all_experiments(tmp_path):
    # create dataset
    experiments = ['exp{}'.format(i) for i in range(5)]
    tissues = ['tissue1', 'tissue2', 'tissue3', 'tissue4', 'tissue5']
    platforms = ['platform5', 'platform4', 'platform3', 'platform2', 'platform1']
    npz_num = [2, 2, 4, 6, 8]
    _create_test_dataset(tmp_path, experiments=experiments, tissues=tissues,
                         platforms=platforms, npz_num=npz_num)

    total_img_num = np.sum(npz_num) * 10

    # initialize db
    db = DatasetBuilder(tmp_path)
    db._identify_tissue_and_platform_types()

    train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

    db._load_all_experiments(data_split=[train_ratio, val_ratio, test_ratio], seed=None)

    # get outputs
    train_dict, val_dict, test_dict = db.train_dict, db.val_dict, db.test_dict

    # check that splits were performed correctly
    for ratio, dict in zip((train_ratio, val_ratio, test_ratio),
                           (train_dict, val_dict, test_dict)):

        X_data, y_data = dict['X'], dict['y']
        assert X_data.shape[0] == ratio * total_img_num
        assert y_data.shape[0] == ratio * total_img_num

        tissue_list, platform_list = dict['tissue_list'], dict['platform_list']
        assert len(tissue_list) == len(platform_list) == X_data.shape[0]

    # check that the metadata maps to the correct images
    for dict in (train_dict, val_dict, test_dict):
        X_data, tissue_list, platform_list = dict['X'], dict['tissue_list'], dict['platform_list']

        # loop over each tissue type, and check that the NPZ is filled with correct constant value
        for constant_val, tissue in enumerate(tissues):

            # index of images with matching tissue type
            tissue_idx = tissue_list == tissue

            images = X_data[tissue_idx]
            assert np.all(images == constant_val)

        # loop over each platform type, and check that the NPZ contains correct constant value
        for constant_val, platform in enumerate(platforms):

            # index of images with matching platform type
            platform_idx = platform_list == platform

            images = X_data[platform_idx]
            assert np.all(images == constant_val)


def test__subset_data_dict(tmp_path):
    # workaround so that __init__ doesn't throw an error
    os.makedirs(os.path.join(tmp_path, 'folder1'))

    X = np.arange(100)
    y = np.arange(100)
    tissue_list = ['tissue1'] * 10 + ['tissue2'] * 50 + ['tissue3'] * 40
    platform_list = ['platform1'] * 20 + ['platform2'] * 40 + ['platform3'] * 40
    data_dict = {'X': X, 'y': y, 'tissue_list': tissue_list, 'platform_list': platform_list}

    db = DatasetBuilder(tmp_path)

    # all tissues, one platform
    tissues = ['tissue1', 'tissue2', 'tissue3']
    platforms = ['platform1']
    subset_dict = db._subset_data_dict(data_dict=data_dict, tissues=tissues, platforms=platforms)
    X_subset = subset_dict['X']
    keep_idx = np.isin(platform_list, platforms)

    assert np.all(X_subset == X[keep_idx])

    # all platforms, one tissue
    tissues = ['tissue2']
    platforms = ['platform1', 'platform2', 'platform3']
    subset_dict = db._subset_data_dict(data_dict=data_dict, tissues=tissues, platforms=platforms)
    X_subset = subset_dict['X']
    keep_idx = np.isin(tissue_list, tissues)

    assert np.all(X_subset == X[keep_idx])

    # drop tissue 1 and platform 3
    tissues = ['tissue2', 'tissue3']
    platforms = ['platform1', 'platform2']
    subset_dict = db._subset_data_dict(data_dict=data_dict, tissues=tissues, platforms=platforms)
    X_subset = subset_dict['X']
    platform_keep_idx = np.isin(platform_list, platforms)
    tissue_keep_idx = np.isin(tissue_list, tissues)
    keep_idx = np.logical_and(platform_keep_idx, tissue_keep_idx)

    assert np.all(X_subset == X[keep_idx])

    # tissue/platform combination that doesn't exist
    tissues = ['tissue1']
    platforms = ['platform3']
    with pytest.raises(ValueError):
        _ = db._subset_data_dict(data_dict=data_dict, tissues=tissues, platforms=platforms)


def test__reshape_dict_no_resize(tmp_path):
    # workaround so that __init__ doesn't throw an error
    os.makedirs(os.path.join(tmp_path, 'folder1'))
    db = DatasetBuilder(tmp_path)

    # create dict
    tissues = ['tissue1', 'tissue2', 'tissue3']
    platforms = ['platform1', 'platform2', 'platform3']
    data_dict = _create_test_dict(tissues=tissues, platforms=platforms)

    # this is 1/2 the size on each dimension as original, so we expect 4x more crops
    output_shape = (20, 20)

    reshaped_dict = db._reshape_dict(dict=data_dict, resize=False, output_shape=output_shape)
    X_reshaped, tissue_list_reshaped = reshaped_dict['X'], reshaped_dict['tissue_list']
    assert X_reshaped.shape[1:3] == output_shape

    # make sure that for each tissue, the arrays with correct value have correct tissue label
    for constant_val, tissue in enumerate(tissues):
        tissue_idx = X_reshaped[:, 0, 0, 0] == constant_val
        tissue_labels = np.array(tissue_list_reshaped)[tissue_idx]
        assert np.all(tissue_labels == tissue)


def test__reshape_dict_by_tissue(tmp_path, mocker):
    mocker.patch('caliban_toolbox.DatasetBuilder.compute_cell_size', mocked_compute_cell_size)
    # workaround so that __init__ doesn't throw an error
    os.makedirs(os.path.join(tmp_path, 'folder1'))
    db = DatasetBuilder(tmp_path)

    # create dict
    tissues = ['tissue1', 'tissue2', 'tissue3']
    platforms = ['platform1', 'platform2', 'platform3']
    data_dict = _create_test_dict(tissues=tissues, platforms=platforms)

    # same size as input data
    output_shape = (40, 40)

    reshaped_dict = db._reshape_dict(dict=data_dict, resize='by_tissue', output_shape=output_shape)
    X_reshaped, tissue_list_reshaped = reshaped_dict['X'], reshaped_dict['tissue_list']
    assert X_reshaped.shape[1:3] == output_shape

    # make sure that for each tissue, the arrays with correct value have correct tissue label
    for constant_val, tissue in enumerate(tissues):
        tissue_idx = X_reshaped[:, 0, 0, 0] == constant_val
        tissue_labels = np.array(tissue_list_reshaped)[tissue_idx]
        assert np.all(tissue_labels == tissue)

        # Each tissue type starts with length 5, and is resized according to its constant value
        assert len(tissue_labels) == 5 * ((constant_val + 1) ** 2)


def test__reshape_dict_by_image(tmp_path, mocker):
    mocker.patch('caliban_toolbox.DatasetBuilder.compute_cell_size', mocked_compute_cell_size)
    # workaround so that __init__ doesn't throw an error
    os.makedirs(os.path.join(tmp_path, 'folder1'))
    db = DatasetBuilder(tmp_path)

    # create dict
    tissues = ['tissue1', 'tissue2', 'tissue3']
    platforms = ['platform1', 'platform2', 'platform3']
    data_dict = _create_test_dict(tissues=tissues, platforms=platforms)

    # same size as input data
    output_shape = (40, 40)

    reshaped_dict = db._reshape_dict(dict=data_dict, resize='by_image', output_shape=output_shape)
    X_reshaped, tissue_list_reshaped = reshaped_dict['X'], reshaped_dict['tissue_list']
    assert X_reshaped.shape[1:3] == output_shape

    # make sure that for each tissue, the arrays with correct value have correct tissue label
    for constant_val, tissue in enumerate(tissues):
        tissue_idx = X_reshaped[:, 0, 0, 0] == constant_val
        tissue_labels = np.array(tissue_list_reshaped)[tissue_idx]
        assert np.all(tissue_labels == tissue)

        # Each tissue type starts with length 5, and is resized according to its constant value
        assert len(tissue_labels) == 5 * ((constant_val + 1) ** 2)


def test_build_dataset(tmp_path):
    # create dataset
    experiments = ['exp{}'.format(i) for i in range(5)]
    tissues = ['tissue1', 'tissue2', 'tissue3', 'tissue4', 'tissue5']
    platforms = ['platform5', 'platform4', 'platform3', 'platform2', 'platform1']
    npz_num = [2, 2, 4, 6, 8]
    _create_test_dataset(tmp_path, experiments=experiments, tissues=tissues,
                         platforms=platforms, npz_num=npz_num)

    db = DatasetBuilder(tmp_path)

    # dataset with all data included
    output_dicts = db.build_dataset(tissues=tissues, platforms=platforms, output_shape=(20, 20))

    for dict in output_dicts:
        # make sure correct tissues and platforms loaded
        current_tissues = dict['tissue_list']
        current_platforms = dict['platform_list']
        assert set(current_tissues) == set(tissues)
        assert set(current_platforms) == set(platforms)

    # dataset with only a subset included
    tissues, platforms = tissues[:3], platforms[:3]
    output_dicts = db.build_dataset(tissues=tissues, platforms=platforms, output_shape=(20, 20))

    for dict in output_dicts:
        # make sure correct tissues and platforms loaded
        current_tissues = dict['tissue_list']
        current_platforms = dict['platform_list']
        assert set(current_tissues) == set(tissues)
        assert set(current_platforms) == set(platforms)

    # cropping to 1/2 the size, there should be 4x more crops
    output_dicts_crop = db.build_dataset(tissues=tissues, platforms=platforms,
                                         output_shape=(10, 10))

    for base_dict, crop_dict in zip(output_dicts, output_dicts_crop):
        X_base, X_crop = base_dict['X'], crop_dict['X']
        assert X_base.shape[0] * 4 == X_crop.shape[0]
