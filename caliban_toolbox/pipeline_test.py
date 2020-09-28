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
import tempfile

import numpy as np
import xarray as xr

from caliban_toolbox import pipeline


def _make_npz_files(npz_num):
    npz_list = []
    for i in range(npz_num):
        X_data = np.zeros((1, 256, 256, 2))
        y_data = np.zeros((1, 256, 256, 1))
        npz_list.append({'X': X_data, 'y': y_data})

    return npz_list


def test_find_sparse_images():
    images = np.zeros((10, 30, 30, 1))
    sparse_indices = np.random.choice(range(10), 5, replace=False)
    non_sparse_mask = ~np.isin(range(10), sparse_indices)

    for idx in range(images.shape[0]):
        img = images[idx]
        if idx in sparse_indices:
            img[0, 0, 0] = 1
            img[0, 5, 0] = 2
        else:
            img[0, :, 0] = range(30)

    pred_indices = pipeline.find_sparse_images(images, 5)

    assert np.all(non_sparse_mask == pred_indices)


def test_save_stitched_npzs():
    channels = np.zeros((4, 100, 100, 2))
    labels = np.zeros((4, 100, 100, 1))

    coords = [['fov1', 'fov2', 'fov3', 'fov4'], range(100), range(100), [0]]
    dims = ['fovs', 'rows', 'cols', 'channels']
    labels = xr.DataArray(labels, coords=coords, dims=dims)

    with tempfile.TemporaryDirectory() as temp_dir:
        pipeline.save_stitched_npzs(channels, labels, temp_dir)
        npzs = os.listdir(temp_dir)
        npz_names = [fov + '.npz' for fov in labels.fovs.values]
        assert np.all(np.isin(npzs, npz_names))


def test_concatenate_npz_files():
    npz_num = 5
    npz_list = _make_npz_files(npz_num=npz_num)
    X_concat, y_concat = pipeline.concatenate_npz_files(npz_list)

    assert X_concat.shape == (npz_num, 256, 256, 2)
    assert y_concat.shape == (npz_num, 256, 256, 1)


def test_create_combined_npz():
    npz_list = _make_npz_files(5)

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, npz in enumerate(npz_list):
            np.savez(os.path.join(temp_dir, 'test_npz_{}.npz'.format(idx)),
                     X=npz['X'], y=npz['y'])

        pipeline.create_combined_npz(npz_dir=temp_dir, save_name='combined.npz')

        combined = np.load(os.path.join(temp_dir, 'combined.npz'))

        assert combined['X'].shape == (5, 256, 256, 2)
        assert combined['y'].shape == (5, 256, 256, 1)
