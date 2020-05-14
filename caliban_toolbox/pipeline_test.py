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
import pandas as pd
import xarray as xr

from caliban_toolbox import pipeline
import importlib
importlib.reload(pipeline)


def _make_raw_metadata():
    metadata_file = {'PROJECT_ID': np.random.randint(1, 100),
                     'EXPERIMENT_ID': np.random.randint(1, 100)}

    return metadata_file


def _make_fov_ids(num_fovs):
    all_fovs = np.random.randint(low=1, high=num_fovs*10, size=num_fovs)
    fovs = ['fov_{}'.format(i) for i in all_fovs]

    return fovs


def _make_exp_metadata(num_fovs):
    fovs = _make_fov_ids(num_fovs)
    raw_metadata = _make_raw_metadata()

    metadata = pd.DataFrame({'image_name': fovs, 'EXPERIMENT_ID': raw_metadata['EXPERIMENT_ID'],
                             'status': 'awaiting_prediction', 'job_name': 'NA'})

    return metadata


def test_create_experiment_folder():
    image_names = _make_fov_ids(10)
    metadata = _make_raw_metadata()

    with tempfile.TemporaryDirectory() as temp_dir:
        experiment_folder = pipeline.create_experiment_folder(image_names=image_names,
                                                              raw_metadata=metadata,
                                                              base_dir=temp_dir)

        saved_metadata = pd.read_csv(os.path.join(experiment_folder, 'metadata.csv'))

        assert np.all(np.isin(saved_metadata['image_name'], image_names))
        assert saved_metadata.loc[0, 'EXPERIMENT_ID'] == metadata['EXPERIMENT_ID']


def test_create_job_folder():
    metadata = _make_exp_metadata(10)
    fov_names = metadata['image_name'].values
    fov_data = np.zeros((len(fov_names), 20, 20, 3))
    fov_num = 7

    with tempfile.TemporaryDirectory() as temp_dir:
        pipeline.create_job_folder(temp_dir, metadata, fov_data, fov_names, fov_num)

        saved_metadata = pd.read_csv(os.path.join(temp_dir, 'metadata.csv'))
        new_status = saved_metadata.loc[np.isin(saved_metadata.image_name, fov_names[:fov_num]), 'status']

        assert np.all(np.isin(new_status, 'in_progress'))


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


def test_process_stitched_data():
    channels = np.zeros((4, 100, 100, 2))
    labels = np.zeros((4, 100, 100, 1))

    coords_labels = [['fov1', 'fov2', 'fov3', 'fov4'], range(100), range(100), [0]]
    coords_channels = [['fov1', 'fov2', 'fov3', 'fov4'], range(100), range(100), range(2)]
    dims = ['fovs', 'rows', 'cols', 'channels']
    labels = xr.DataArray(labels, coords=coords_labels, dims=dims)
    channels = xr.DataArray(channels, coords=coords_channels, dims=dims)

    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, 'output'))
        labels.to_netcdf(os.path.join(temp_dir, 'output', 'stitched_labels.xr'))
        channels.to_netcdf(os.path.join(temp_dir, 'channel_data.xr'))

        pipeline.process_stitched_data(temp_dir)