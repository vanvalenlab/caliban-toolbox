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


from caliban_toolbox.dataset_builder import train_val_test_split, DatasetBuilder


def _create_test_npz(path, constant_value=1, X_shape=(10, 20, 20, 3), y_shape=(10, 20, 20, 1)):
    X_data = np.full(X_shape, constant_value)
    y_data = np.full(y_shape, constant_value)
    np.savez(path, X=X_data, y=y_data)


def _create_test_dataset(path, experiment_list, tissue_list, platform_list, npz_num):
    """Creates an example directory to load data from

    Args:
        path: folder to hold datasets
        experiment_list: list of experiment names
        tissue_list: list of tissue types for each experiment
        platform_list: list of platform types for each experiment
        npz_num: number of unique NPZ files within each experiment. The NPZs within
            each experiment are constant-valued arrays corresponding to the index of that exp

    Raises:
        ValueError: If tissue_list, platform_list, or NPZ_num have different lengths
    """
    lengths = [len(x) for x in [experiment_list, tissue_list, platform_list, npz_num]]
    if len(set(lengths)) != 1:
        raise ValueError('All inputs must have the same length')

    for i in range(len(experiment_list)):
        experiment_folder = os.path.join(path, experiment_list[i])
        os.makedirs(experiment_folder)

        metadata = dict()
        metadata['tissue'] = tissue_list[i]
        metadata['platform'] = platform_list[i]

        metadata_path = os.path.join(experiment_folder, 'metadata.json')

        with open(metadata_path, 'w') as write_file:
            json.dump(metadata, write_file)

        for npz in range(npz_num[i]):
            _create_test_npz(path=os.path.join(experiment_folder, 'sub_exp_{}.npz'.format(npz)),
                             constant_value=i)


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


def test__create_tissue_and_platform_dict(tmp_path):
    # create dataset
    experiment_list = ['exp{}'.format(i) for i in range(5)]
    tissue_list = ['tissue1', 'tissue2', 'tissue3', 'tissue2', 'tissue1']
    platform_list = ['platform1', 'platform1', 'platform2', 'platform2', 'platform3']
    npz_num = [1] * 5
    _create_test_dataset(tmp_path, experiment_list=experiment_list, tissue_list=tissue_list,
                         platform_list=platform_list, npz_num=npz_num)

    db = DatasetBuilder(dataset_path=tmp_path)

    db._create_tissue_and_platform_dict()

    # check that all tissues and platforms added
    assert set(db.all_tissues) == set(tissue_list)
    assert set(db.all_platforms) == set(platform_list)

    # check that each entry in dict correctly maps to corresponding entry in reverse dict
    for tissue in set(tissue_list):
        tissue_id = db.tissue_dict[tissue]
        reversed_tissue = db.rev_tissue_dict[tissue_id]

        assert tissue == reversed_tissue

    for platform in set(platform_list):
        platform_id = db.platform_dict[platform]
        reversed_platform = db.rev_platform_dict[platform_id]

        assert platform == reversed_platform


def test__load_experiment_single_npz(tmp_path):
    exp_list, tissue_list, platform_list, npz_num = ['exp1'], ['tissue1'], ['platform1'], [1]
    _create_test_dataset(tmp_path, experiment_list=exp_list, tissue_list=tissue_list,
                         platform_list=platform_list, npz_num=npz_num)

    # initialize db
    db = DatasetBuilder(tmp_path)

    # load dataset
    X, y, tissue, platform = db._load_experiment(os.path.join(tmp_path, exp_list[0]))

    # A single NPZ with 10 images
    assert X.shape[0] == 10
    assert y.shape[0] == 10

    assert tissue == tissue_list[0]
    assert platform == platform_list[0]


def test__load_experiment_multiple_npz(tmp_path):
    exp_list, tissue_list, platform_list, npz_num = ['exp1'], ['tissue1'], ['platform1'], [5]
    _create_test_dataset(tmp_path, experiment_list=exp_list, tissue_list=tissue_list,
                         platform_list=platform_list, npz_num=npz_num)

    # initialize db
    db = DatasetBuilder(tmp_path)

    # load dataset
    X, y, tissue, platform = db._load_experiment(os.path.join(tmp_path, exp_list[0]))

    # 5 NPZs with 10 images each
    assert X.shape[0] == 50
    assert y.shape[0] == 50

    assert tissue == tissue_list[0]
    assert platform == platform_list[0]


def test__load_all_experiments(tmp_path):
    # create dataset
    experiment_list = ['exp{}'.format(i) for i in range(5)]
    tissue_list = ['tissue1', 'tissue2', 'tissue3', 'tissue4', 'tissue5']
    platform_list = ['platform5', 'platform4', 'platform3', 'platform2', 'platform1']
    npz_num = [2, 2, 4, 6, 8]
    _create_test_dataset(tmp_path, experiment_list=experiment_list, tissue_list=tissue_list,
                         platform_list=platform_list, npz_num=npz_num)

    total_img_num = np.sum(npz_num) * 10

    # initialize db
    db = DatasetBuilder(tmp_path)
    db._create_tissue_and_platform_dict()

    train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

    db._load_all_experiments(data_split=[train_ratio, val_ratio, test_ratio], seed=None)

    # get outputs
    train_dict, val_dict, test_dict = db.train_dict, db.val_dict, db.test_dict
    tissue_dict, platform_dict = db.tissue_dict, db.platform_dict

    # check that splits were performed correctly
    for ratio, dict in zip((train_ratio, val_ratio, test_ratio),
                           (train_dict, val_dict, test_dict)):

        X_data, y_data = dict['X'], dict['y']
        assert X_data.shape[0] == ratio * total_img_num
        assert y_data.shape[0] == ratio * total_img_num

        tissue_id, platform_id = dict['tissue_id'], dict['platform_id']
        assert len(tissue_id) == len(platform_id) == X_data.shape[0]

    # check that the metadata maps to the correct images
    for dict in (train_dict, val_dict, test_dict):
        X_data, tissue_ids, platform_ids = dict['X'], dict['tissue_id'], dict['platform_id']

        # loop over each tissue type, and check that the NPZ is filled with correct constant value
        for constant_val, tissue_name in enumerate(tissue_list):
            tissue_id = tissue_dict[tissue_name]

            # index of images with matching tissue type
            tissue_idx = tissue_ids == tissue_id

            images = X_data[tissue_idx]
            assert np.all(images == constant_val)

        # loop over each platform type, and check that the NPZ is filled with correct constant value
        for constant_val, platform_name in enumerate(platform_list):
            platform_id = platform_dict[platform_name]

            # index of images with matching platform type
            platform_idx = platform_ids == platform_id

            images = X_data[platform_idx]
            assert np.all(images == constant_val)


def test__subset_data_dict(tmp_path):
    # workaround so that __init__ doesn't throw an error
    os.makedirs(os.path.join(tmp_path, 'folder1'))

    X = np.arange(100)
    y = np.arange(100)
    tissue_id = np.concatenate((np.repeat(1, 10), np.repeat(2, 50), np.repeat(3, 40)), axis=0)
    platform_id = np.concatenate((np.repeat(1, 20), np.repeat(2, 40), np.repeat(3, 40)))
    data_dict = {'X': X, 'y': y, 'tissue_id': tissue_id, 'platform_id': platform_id}

    tissue_dict = {'tissue{}'.format(i): i for i in range(1, 4)}
    platform_dict = {'platform{}'.format(i): i for i in range(1, 4)}

    db = DatasetBuilder(tmp_path)
    db.tissue_dict = tissue_dict
    db.platform_dict = platform_dict

    # all tissues, one platform
    tissues = ['tissue1', 'tissue2', 'tissue3']
    platforms = ['platform1']
    subset_dict = db._subset_data_dict(data_dict=data_dict, tissues=tissues, platforms=platforms)
    X_subset = subset_dict['X']
    keep_idx = platform_id == 1

    assert np.all(X_subset == X[keep_idx])

    # all platforms, one tissue
    tissues = ['tissue2']
    platforms = ['platform1', 'platform2', 'platform3']
    subset_dict = db._subset_data_dict(data_dict=data_dict, tissues=tissues, platforms=platforms)
    X_subset = subset_dict['X']
    keep_idx = tissue_id == 2

    assert np.all(X_subset == X[keep_idx])

    # drop tissue 1 and platform 3
    tissues = ['tissue2', 'tissue3']
    platforms = ['platform1', 'platform2']
    subset_dict = db._subset_data_dict(data_dict=data_dict, tissues=tissues, platforms=platforms)
    X_subset = subset_dict['X']
    platform_keep_idx = platform_id != 3
    tissue_keep_idx = tissue_id != 1
    keep_idx = np.logical_and(platform_keep_idx, tissue_keep_idx)

    assert np.all(X_subset == X[keep_idx])

    # tissue/platform combination that doesn't exist
    tissues = ['tissue1']
    platforms = ['platform3']
    with pytest.raises(ValueError):
        _ = db._subset_data_dict(data_dict=data_dict, tissues=tissues, platforms=platforms)


def test_build_dataset(tmp_path):
    # create dataset
    experiment_list = ['exp{}'.format(i) for i in range(5)]
    tissue_list = ['tissue1', 'tissue2', 'tissue3', 'tissue4', 'tissue5']
    platform_list = ['platform5', 'platform4', 'platform3', 'platform2', 'platform1']
    npz_num = [2, 2, 4, 6, 8]
    _create_test_dataset(tmp_path, experiment_list=experiment_list, tissue_list=tissue_list,
                         platform_list=platform_list, npz_num=npz_num)

    db = DatasetBuilder(tmp_path)

    # dataset with all data included
    output_dicts = db.build_dataset(tissues=tissue_list, platforms=platform_list)
    tissue_ids = [db.tissue_dict[tissue] for tissue in tissue_list]
    platform_ids = [db.platform_dict[platform] for platform in platform_list]

    for dict in output_dicts:
        # make sure correct tissues and platforms loaded
        current_tissues = dict['tissue_id']
        current_platforms = dict['platform_id']
        assert set(current_tissues) == set(tissue_ids)
        assert set(current_platforms) == set(platform_ids)


    # dataset with only a subset included
    tissue_list, tissue_ids = tissue_list[:3], tissue_ids[:3]
    platform_list, platform_ids = platform_list[:3], platform_ids[:3]
    output_dicts = db.build_dataset(tissues=tissue_list, platforms=platform_list)

    for dict in output_dicts:
        # make sure correct tissues and platforms loaded
        current_tissues = dict['tissue_id']
        current_platforms = dict['platform_id']
        assert set(current_tissues) == set(tissue_ids)
        assert set(current_platforms) == set(platform_ids)


def test_train_val_test_split():
    X_data = np.zeros((100, 5, 5, 3))
    y_data = np.zeros((100, 5, 5, 1))

    train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

    X_train, y_train, X_val, y_val, X_test, y_test, = train_val_test_split(X_data=X_data,
                                                                           y_data=y_data,
                                                                           train_ratio=train_ratio,
                                                                           val_ratio=val_ratio,
                                                                           test_ratio=test_ratio)

    assert X_train.shape[0] == 100 * train_ratio
    assert y_train.shape[0] == 100 * train_ratio

    assert X_val.shape[0] == 100 * val_ratio
    assert y_val.shape[0] == 100 * val_ratio

    assert X_test.shape[0] == 100 * test_ratio
    assert y_test.shape[0] == 100 * test_ratio
