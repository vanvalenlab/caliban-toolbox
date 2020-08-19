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


def _create_test_npz(path, X_shape=(2, 20, 20, 3), y_shape=(2, 20, 20, 1)):
    X_data = np.zeros(X_shape)
    y_data = np.zeros(y_shape)
    np.savez(path, X=X_data, y=y_data)


def _create_test_dataset(path, experiment_list, tissue_list, platform_list, npz_num):
    """Creates an example directory to load data from

    Args:
        path: folder to hold datasets
        experiment_list: list of experiment names
        tissue_list: list of tissue types for each experiment
        platform_list: list of platform types for each experiment
        npz_num: number of unique NPZ files within each experiment

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
            _create_test_npz(os.path.join(experiment_folder, 'sub_exp_{}.npz'.format(npz)))


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

    # A single NPZ with 2 images
    assert X.shape[0] == 2
    assert y.shape[0] == 2

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

    # 5 NPZs with 2 images each
    assert X.shape[0] == 10
    assert y.shape[0] == 10

    assert tissue == tissue_list[0]
    assert platform == platform_list[0]




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
