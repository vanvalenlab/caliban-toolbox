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

import numpy as np


from caliban_toolbox.dataset_builder import train_val_test_split, DatasetBuilder


def create_npz(path, X_shape=(5, 20, 20, 3), y_shape=(5, 20, 20, 1)):
    X_data = np.zeros(X_shape)
    y_data = np.zeros(y_shape)
    np.savez(path, X=X_data, y=y_data)


def create_dataset_dir(path, tissue_list, platform_list, npz_num):
    """Creates an example directory to load data from

    Args:
        path: folder to hold datasets
        tissue_list: list of tissue types for each experiment
        platform_list: list of platform types for each experiment
        npz_num: number of unique NPZ files within each experiment

    Raises:
        ValueError: If tissue_list, platform_list, or NPZ_num have different lengths
    """

    if len(tissue_list) != len(platform_list) or len(tissue_list) != len(npz_num):
        raise ValueError('All inputs must have the same length')

    exp_names = ['experiment_{}'.format(i) for i in range(len(platform_list))]

    for i in range(len(exp_names)):
        experiment_folder = os.path.join(path, exp_names[i])
        os.makedirs(experiment_folder)

        metadata = dict()
        metadata['tissue'] = tissue_list[i]
        metadata['platform'] = platform_list[i]

        metadata_path = os.path.join(experiment_folder, 'metadata.json')

        with open(metadata_path, 'w') as write_file:
            json.dump(metadata, write_file)

        for npz in range(len(npz_num)):
            create_npz(os.path.join(experiment_folder, 'sub_exp_{}.npz'.format(npz)))


def test__init__(tmp_path):
    resize = False
    db = DatasetBuilder(dataset_path=tmp_path, resize=resize)

    assert db.dataset_path == tmp_path
    assert db.resize == resize


def test__create_tissue_and_platform_dict(tmp_path):
    # create dataset
    tissue_list = ['tissue1', 'tissue2', 'tissue3', 'tissue2', 'tissue1']
    platform_list = ['platform1', 'platform1', 'platform2', 'platform2', 'platform3']
    npz_num = [1] * 5
    create_dataset_dir(tmp_path, tissue_list=tissue_list, platform_list=platform_list,
                       npz_num=npz_num)

    db = DatasetBuilder(dataset_path=tmp_path)

    db._create_tissue_and_platform_dict()

    # check that all tissues and platforms added
    assert db.all_tissues == set(tissue_list)
    assert db.all_platforms == set(platform_list)

    # check that each entry in dict correctly maps to corresponding entry in reverse dict
    for tissue in set(tissue_list):
        tissue_id = db.tissue_dict[tissue]
        reversed_tissue = db.rev_tissue_dict[tissue_id]

        assert tissue == reversed_tissue

    for platform in set(platform_list):
        platform_id = db.platform_dict[platform]
        reversed_platform = db.rev_platform_dict[platform_id]

        assert platform == reversed_platform


def test_train_val_test_split():
    X_data = np.zeros((100, 5, 5, 3))
    y_data = np.zeros((100, 5, 5, 1))

    train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

    train_split, val_split, test_split = train_val_test_split(X_data=X_data,
                                                              y_data=y_data,
                                                              train_ratio=train_ratio,
                                                              val_ratio=val_ratio,
                                                              test_ratio=test_ratio)

    assert train_split[0].shape[0] == 100 * train_ratio
    assert train_split[1].shape[0] == 100 * train_ratio

    assert val_split[0].shape[0] == 100 * val_ratio
    assert val_split[1].shape[0] == 100 * val_ratio

    assert test_split[0].shape[0] == 100 * test_ratio
    assert test_split[1].shape[0] == 100 * test_ratio