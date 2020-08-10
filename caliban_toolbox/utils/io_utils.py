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
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import os
import json

from itertools import product


def save_npzs_for_caliban(X_data, y_data, original_data, log_data, save_dir,
                          blank_labels='include', save_format='npz', verbose=True):
    """Take an array of processed image data and save as NPZ for caliban

    Args:
        X_data: 7D tensor of cropped and sliced raw images
        y_data: 7D tensor of cropped and sliced labeled images
        original_data: the original unmodified images
        log_data: data used to reconstruct images
        save_dir: path to save the npz and JSON files
        blank_labels: whether to include NPZs with blank labels (poor predictions)
                      or skip (no cells)
        save_format: format to save the data (currently only NPZ)
        verbose: flag to control print statements
    """

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # if these are present, it means data was cropped/sliced. Otherwise, default to 1
    num_crops = log_data.get('num_crops', 1)
    num_slices = log_data.get('num_slices', 1)

    fov_names = original_data.fovs.values
    fov_len = len(fov_names)

    if blank_labels not in ['skip', 'include', 'separate']:
        raise ValueError('blank_labels must be one of '
                         '[skip, include, separate], got {}'.format(blank_labels))

    if blank_labels == 'separate':
        os.makedirs(os.path.join(save_dir, 'separate'))

    # for each fov, loop through 2D crops and 3D slices
    for fov, crop, slice in product(range(fov_len), range(num_crops), range(num_slices)):
        # generate identifier for crop
        npz_id = 'fov_{}_crop_{}_slice_{}'.format(fov_names[fov], crop, slice)

        # get working batch
        labels = y_data[fov, :, crop, slice, ...].values
        channels = X_data[fov, :, crop, slice, ...].values

        # determine if labels are blank, and if so what to do with npz
        if np.sum(labels) == 0:

            # blank labels get saved to separate folder
            if blank_labels == 'separate':
                if verbose:
                    print('{} is blank, saving to separate folder'.format(npz_id))
                save_path = os.path.join(save_dir, blank_labels, npz_id)

                # save images as either npz or xarray
                if save_format == 'npz':
                    np.savez(save_path + '.npz', X=channels, y=labels)

                elif save_format == 'xr':
                    raise NotImplementedError()

            # blank labels don't get saved, empty area of tissue
            elif blank_labels == 'skip':
                if verbose:
                    print('{} is blank, skipping saving'.format(npz_id))

            # blank labels get saved along with other crops
            elif blank_labels == 'include':
                if verbose:
                    print('{} is blank, saving to folder'.format(npz_id))
                save_path = os.path.join(save_dir, npz_id)

                # save images as either npz or xarray
                if save_format == 'npz':
                    np.savez(save_path + '.npz', X=channels, y=labels)

                elif save_format == 'xr':
                    raise NotImplementedError()

        else:
            # crop is not blank, save based on file_format
            save_path = os.path.join(save_dir, npz_id)

            # save images as either npz or xarray
            if save_format == 'npz':
                np.savez(save_path + '.npz', X=channels, y=labels)

            elif save_format == 'xr':
                raise NotImplementedError()

    log_data['fov_names'] = fov_names.tolist()
    log_data['label_name'] = str(y_data.coords[y_data.dims[-1]][0].values)
    log_data['original_shape'] = original_data.shape
    log_data['slice_stack_len'] = X_data.shape[1]
    log_data['save_format'] = save_format

    log_path = os.path.join(save_dir, 'log_data.json')
    with open(log_path, 'w') as write_file:
        json.dump(log_data, write_file)


def get_saved_file_path(dir_list, fov_name, crop, slice, file_ext='.npz'):
    """Helper function to identify correct file path for an npz file

    Args:
        dir_list: list of files in directory
        fov_name: string of the current fov_name
        crop: int of current crop
        slice: int of current slice
        file_ext: extension file was saved with

    Returns:
        string: formatted file name

    Raises:
        ValueError: If multiple file path matches were found
    """

    base_string = 'fov_{}_crop_{}_slice_{}'.format(fov_name, crop, slice)
    string_matches = [string for string in dir_list if base_string + '_save_version' in string]

    if len(string_matches) == 0:
        full_string = base_string + file_ext
    elif len(string_matches) == 1:
        full_string = string_matches[0]
    else:
        raise ValueError('Multiple save versions found: '
                         'please select only a single save version. {}'.format(string_matches))
    return full_string


def load_npzs(crop_dir, log_data, verbose=True):
    """Reads all of the cropped images from a directory, and aggregates them into a single stack

    Args:
        crop_dir: path to directory with cropped npz or xarray files
        log_data: dictionary of parameters generated during data saving

        verbose: flag to control print statements

    Returns:
        numpy.array: 7D tensor of labeled crops
    """

    fov_names = log_data['fov_names']
    fov_len, stack_len, _, _, row_size, col_size, _ = log_data['original_shape']
    save_format = log_data['save_format']

    # if cropped/sliced, get size of dimensions. Otherwise, use size in original data
    row_crop_size = log_data.get('row_crop_size', row_size)
    col_crop_size = log_data.get('col_crop_size', col_size)
    slice_stack_len = log_data.get('slice_stack_len', stack_len)

    # if cropped/sliced, get number of crops/slices
    num_crops, num_slices = log_data.get('num_crops', 1), log_data.get('num_slices', 1)
    stack = np.zeros((fov_len, slice_stack_len, num_crops,
                      num_slices, row_crop_size, col_crop_size, 1))
    saved_files = os.listdir(crop_dir)

    # for each fov, loop over each 2D crop and 3D slice
    for fov, crop, slice in product(range(fov_len), range(num_crops), range(num_slices)):
        # load NPZs
        if save_format == 'npz':
            npz_path = os.path.join(crop_dir, get_saved_file_path(saved_files,
                                                                  fov_names[fov],
                                                                  crop, slice))
            if os.path.exists(npz_path):
                temp_npz = np.load(npz_path)

                # determine how labels were named
                labels_key = 'y' if 'y' in temp_npz else 'annotated'

                # last slice may be truncated, modify index
                if slice == num_slices - 1:
                    current_stack_len = temp_npz[labels_key].shape[1]
                else:
                    current_stack_len = slice_stack_len

                stack[fov, :current_stack_len, crop, slice, ...] = temp_npz[labels_key]
            else:
                # npz not generated, did not contain any labels, keep blank
                if verbose:
                    print('could not find npz {}, skipping'.format(npz_path))

        # load xarray
        elif save_format == 'xr':
            raise NotImplementedError()
            # xr_path = os.path.join(crop_dir, get_saved_file_path(saved_files, fov_names[fov],
            #                                                      crop, slice))
            # if os.path.exists(xr_path):
            #     temp_xr = xr.open_dataarray(xr_path)
            #
            #     # last slice may be truncated, modify index
            #     if slice == num_slices - 1:
            #         current_stack_len = temp_xr.shape[1]
            #     else:
            #         current_stack_len = stack_len
            #
            #     stack[fov, :current_stack_len, crop, slice, ...] = temp_xr[..., -1:]
            # else:
            #     # npz not generated, did not contain any labels, keep blank
            #     print('could not find xr {}, skipping'.format(xr_path))

    return stack
