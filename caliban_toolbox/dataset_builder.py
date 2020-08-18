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
import errno
import json
import math
import numpy as np

from skimage.segmentation import relabel_sequential

from caliban_toolbox.utils.misc_utils import list_npzs_folder


class DatasetBuilder(object):
    """Class to build a dataset from annotated data

    Args:
        dataset_path: path to dataset. Within the dataset, each unique experiment
            has its own folder with a dedicated metadata file
    """
    def __init__(self,
                 dataset_path=None,
                 resize=False):

        self.dataset_path = dataset_path
        self.resize = resize

        if not os.path.exists(dataset_path):
            raise ValueError('Invalid dataset path supplied')



    def _create_tissue_and_platform_dict(self):
        """Creates a dictionary mapping strings to numeric values for all platforms and tissues"""

        tissues = []
        platforms = []
        for folder in self.dataset_folders:
            file_path = os.path.join(self.dataset_path, folder, 'metadata.json')
            with open(file_path) as f:
                metadata = json.load(f)
            tissues.append(metadata['tissue'])
            platforms.append(metadata['platform'])

        tissues = list(set(tissues))
        platforms = list(set(platforms))
        self.all_tissues = tissues
        self.all_platforms = platforms

        tissue_dict = {}
        rev_tissue_dict = {}
        platform_dict = {}
        rev_platform_dict = {}

        for i, tissue in enumerate(tissues):
            tissue_dict[tissue] = i
            rev_tissue_dict[i] = tissue
        for i, platform in enumerate(platforms):
            platform_dict[platform] = i
            rev_platform_dict[i] = platform

        self.tissue_dict = tissue_dict
        self.rev_tissue_dict = rev_tissue_dict
        self.platform_dict = platform_dict
        self.rev_platform_dict = rev_platform_dict


    def _load_experiment(self, experiment_path):
        """Load the NPZ files present in a single experiment folder

        Args:
            experiment_path: the full path to a folder of NPZ files and metadata file

        Returns:
            tuple of X and y data from all NPZ files in the experiment
            tissue_ids: list of same length as training data with numeric tissue id
            platform_ids: list of same length as training data with numeric platform id
        """

        X_list = []
        y_list = []

        # get all NPZ files present in current experiment directory
        npz_files = list_npzs_folder(experiment_path)
        for file in npz_files:
            npz_path = os.path.join(experiment_path, file)
            training_data = np.load(npz_path)

            X = training_data['X']
            y = training_data['y']

            X_list.append(X)
            y_list.append(y)

        # get associated metadata
        metadata_path = os.path.join(experiment_path, 'metadata.json')
        with open(metadata_path) as f:
            metadata = json.load(f)

        # combine all NPZ files together
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        tissue = metadata['tissue']
        platform = metadata['platform']
        tissue_id = self.tissue_dict[tissue]
        platform_id = self.platform_dict[platform]

        tissue_ids = np.array([tissue_id] * X.shape[0])
        platform_ids = np.array([platform_id] * X.shape[0])

        return X, y, tissue_ids, platform_ids



    def build_dataset(self,
                      tissues='all',
                      platforms='all'):

        self.dataset_folders = os.listdir(dataset_path)
        self._create_tissue_and_platform_dict()

        X_list = []
        y_list = []
        tissue_ids = []
        platform_ids = []

        for folder in self.dataset_folders:
            folder_path = os.path.join(self.dataset_path, folder)
            X, y, tissue_id, platform_id = self._load_dataset(folder_path)
            X_list.append(X)
            y_list.append(y)
            tissue_ids.append(tissue_id)
            platform_ids.append(platform_id)

        self.X = np.concatenate(X_list, axis=0)
        self.y = np.concatenate(y_list, axis=0)
        self.tissue_ids = np.concatenate(tissue_ids, axis=0)
        self.platform_ids = np.concatenate(platform_ids, axis=0)


        if isinstance(tissues, list):
            for tissue in tissues:
                if tissue not in self.all_tissues:
                    raise ValueError('{} is not one of {}'.format(tissue, self.all_tissues))
        elif tissues == 'all':
            tissues = self.all_tissues
        elif tissues in self.all_tissues:
            tissues = [tissues]
        else:
            raise ValueError(
                'tissues should be "all", one of {}, or a list of acceptable tissue types'.format(
                    self.all_tissues))

        if isinstance(platforms, list):
            for platform in platforms:
                if platform not in self.all_platforms:
                    raise ValueError('{} is not one of {}'.format(platform, self.all_platforms))
        elif platforms == 'all':
            platforms = self.all_platforms
        elif platforms in self.all_platforms:
            platforms = [platforms]
        else:
            raise ValueError(
                'platforms should be "all", one of {}, or a list of acceptable platform types'.format(
                    self.all_platforms))

        # Identify locations with the correct tissue types
        tissue_numbers = [self.tissue_dict[tissue] for tissue in tissues]
        tissue_locs = [tid in tissue_numbers for tid in self.tissue_ids]

        # Identify locations with the correct platform types
        platform_numbers = [self.platform_dict[platform] for platform in platforms]
        platform_locs = [pid in platform_numbers for pid in self.platform_ids]

        tissue_locs = np.array(tissue_locs)
        platform_locs = np.array(platform_locs)
        good_locations = tissue_locs * platform_locs

        X_dict = self.X[good_locations]
        y_dict = self.y[good_locations]
        tissue_ids_dict = self.tissue_ids[good_locations]
        platform_ids_dict = self.platform_ids[good_locations]

        train_dict = {}
        train_dict['X'] = X_dict
        train_dict['y'] = y_dict
        train_dict['tissue_ids'] = tissue_ids_dict
        train_dict['platform_ids'] = platform_ids_dict
        train_dict['tissue_dict'] = self.tissue_dict
        train_dict['platform_dict'] = self.platform_dict
        train_dict['rev_tissue_dict'] = self.rev_tissue_dict
        train_dict['rev_platform_dict'] = self.rev_platform_dict

        return train_dict


from sklearn.model_selection import train_test_split


def train_val_test_split(X_data, y_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    """Randomly splits supplied data into specified sizes for model assessment

    Args:
        X_data: array of X data
        y_data: array of y_data
        train_ratio: fraction of dataset for train split
        val_ratio: fraction of dataset for val split
        test_ratio: optional fraction of dataset for test split,
            otherwise only computes a train/val split
        seed: random seed for reproducible splits

    Returns:
        list of X and y data split appropriately

    Raises:
        ValueError: if ratios do not sum to 1
        ValueError: If length of X and y data is not equal
    """

    total = np.round(train_ratio + val_ratio + test_ratio, decimals=2)
    if total != 1:
        raise ValueError('Data splits must sum to 1, supplied splits sum to {}'.format(total))

    if X_data.shape[0] != y_data.shape[0]:
        raise ValueError('Supplied X and y data do not have the same '
                         'length over batches dimension. '
                         'X.shape: {}, y.shape: {}'.format(X_data.shape, y_data.shape))

    # compute fraction not in train
    remainder_size = np.round(1 - train_ratio, decimals=2)

    # split dataset into train and remainder
    X_train, X_remainder, y_train, y_remainder = train_test_split(X_data, y_data,
                                                                  test_size=remainder_size,
                                                                  random_state=seed)
    # check and see if there is a test split
    if test_ratio > 0:

        # compute fraction of remainder that is test
        test_size = np.round(test_ratio / (val_ratio + test_ratio), decimals=2)

        # split remainder into val and test
        X_val, X_test, y_val, y_test = train_test_split(X_remainder, y_remainder,
                                                        test_size=test_size,
                                                        random_state=seed)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    else:
        return (X_train, y_train), (X_remainder, y_remainder)
