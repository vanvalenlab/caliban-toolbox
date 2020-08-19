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
import pytest

import numpy as np

from caliban_toolbox.DatasetBenchmarker import DatasetBenchmarker


def _create_labels(offset=0):
    labels = np.zeros((5, 100, 100, 1))

    base_label = np.zeros((1, 100, 100, 1))
    base_label[0, :20, :20] = 1
    base_label[0, 20:34, 30:50] = 2
    base_label[0, 48:52, 10:20] = 3
    base_label[0, 82:100, 70:90] = 4

    labels[:, offset:, offset:] = base_label[:, :(100 - offset), :(100 - offset)]

    return labels


def test__init__():
    y_true, y_pred = _create_labels(), _create_labels()
    tissue_ids = ['tissue{}'.format(i) for i in range(5)]
    platform_ids = ['platform{}'.format(i) for i in range(5)]
    db = DatasetBenchmarker(y_true=y_true, y_pred=y_pred, tissue_ids=tissue_ids,
                            platform_ids=platform_ids, model_name='test')

    with pytest.raises(ValueError, match='Shape mismatch'):
        _ = DatasetBenchmarker(y_true=y_true, y_pred=y_pred[0], tissue_ids=tissue_ids,
                               platform_ids=platform_ids, model_name='test')

    with pytest.raises(ValueError, match='Data must be 4D'):
        _ = DatasetBenchmarker(y_true=y_true[0], y_pred=y_pred[0], tissue_ids=tissue_ids,
                               platform_ids=platform_ids, model_name='test')

    with pytest.raises(ValueError, match='Tissue_ids and platform_ids'):
        _ = DatasetBenchmarker(y_true=y_true, y_pred=y_pred, tissue_ids=tissue_ids[1:],
                               platform_ids=platform_ids, model_name='test')

    with pytest.raises(ValueError, match='Tissue_ids and platform_ids'):
        _ = DatasetBenchmarker(y_true=y_true, y_pred=y_pred, tissue_ids=tissue_ids,
                               platform_ids=platform_ids[1:], model_name='test')


def test__benchmark_category():
    # perfect agreement
    y_true_category_1, y_pred_category_1 = _create_labels(), _create_labels()

    # small offset between labels
    y_true_category_2, y_pred_category_2 = _create_labels(), _create_labels(offset=3)

    # large offset between labels
    y_true_category_3, y_pred_category_3 = _create_labels(), _create_labels(offset=5)

    y_true = np.concatenate((y_true_category_1, y_true_category_2, y_true_category_3))
    y_pred = np.concatenate((y_pred_category_1, y_pred_category_2, y_pred_category_3))
    tissue_ids = ['tissue1'] * 5 + ['tissue2'] * 5 + ['tissue3'] * 5
    platform_ids = ['platform1'] * 15

    # initialize
    db = DatasetBenchmarker(y_true=y_true, y_pred=y_pred, tissue_ids=tissue_ids,
                            platform_ids=platform_ids, model_name='test')
    db.metrics.calc_object_stats(y_true, y_pred)

    # compute across tissues
    stats_dict = db._benchmark_category(category_ids=tissue_ids)

    assert stats_dict['tissue1']['recall'] == 1
    assert stats_dict['tissue1']['jaccard'] == 1

    assert stats_dict['tissue2']['recall'] > stats_dict['tissue3']['recall']
    assert stats_dict['tissue2']['jaccard'] > stats_dict['tissue3']['jaccard']


def test_benchmark():
    y_true, y_pred = _create_labels(), _create_labels(offset=1)
    tissue_ids = ['tissue1'] * 2 + ['tissue2'] * 3
    platform_ids = ['platform1'] * 3 + ['platform2'] * 2

    db = DatasetBenchmarker(y_true=y_true, y_pred=y_pred, tissue_ids=tissue_ids,
                            platform_ids=platform_ids, model_name='test')

    tissue_stats, platform_stats, all_stats = db.benchmark()

    assert set(tissue_stats.keys()) == set(tissue_ids)
    assert set(platform_stats.keys()) == set(platform_ids)
    assert set(all_stats.keys()) == {'all'}
