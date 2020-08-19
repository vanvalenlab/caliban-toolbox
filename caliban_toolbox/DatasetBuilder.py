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

from skimage.segmentation import relabel_sequential
from skimage.measure import label
from skimage.morphology import remove_small_objects

from caliban_toolbox.utils.misc_utils import list_npzs_folder, list_folders
from caliban_toolbox.build import train_val_test_split, reshape_training_data, compute_cell_size


class DatasetBuilder(object):
    """Class to build a dataset from annotated data

    Args:
        dataset_path: path to dataset. Within the dataset, each unique experiment
            has its own folder with a dedicated metadata file

    Raises:
        ValueError: if invalid dataset_path
        ValueError: if no folders in dataset_path
    """
    def __init__(self,
                 dataset_path):

        self.dataset_path = dataset_path

        if not os.path.exists(dataset_path):
            raise ValueError('Invalid dataset path supplied')

        experiment_folders = list_folders(dataset_path)
        if experiment_folders == []:
            raise ValueError('No experiment folders found in dataset')
        self.experiment_folders = experiment_folders

        self.all_tissues = []
        self.all_platforms = []

        # dicts to hold aggregated data
        self.train_dict = {}
        self.val_dict = {}
        self.test_dict = {}

        # parameters for splitting the data
        self.data_split = None
        self.seed = None

    def _identify_tissue_and_platform_types(self):
        """Identify all of the unique tissues and platforms in the dataset"""

        tissues = []
        platforms = []
        for folder in self.experiment_folders:
            file_path = os.path.join(self.dataset_path, folder, 'metadata.json')
            with open(file_path) as f:
                metadata = json.load(f)
            tissues.append(metadata['tissue'])
            platforms.append(metadata['platform'])

        self.all_tissues = list(set(tissues))
        self.all_platforms = list(set(platforms))

    def _load_experiment(self, experiment_path):
        """Load the NPZ files present in a single experiment folder

        Args:
            experiment_path: the full path to a folder of NPZ files and metadata file

        Returns:
            tuple of X and y data from all NPZ files in the experiment
            tissue: the tissue type of this experiment
            platform: the platform type of this experiment
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

        return X, y, tissue, platform

    def _load_all_experiments(self, data_split, seed):
        """Loads all experiment data from experiment folder to enable dataset building

        Args:
            data_split: tuple specifying the fraction of the dataset for train/val/test
            seed: seed for reproducible splitting of dataset

        Raises:
            ValueError: If any of the NPZ files have different non-batch dimensions
        """
        X_train, X_val, X_test = [], [], []
        y_train, y_val, y_test = [], [], []
        tissue_list_train, tissue_list_val, tissue_list_test = [], [], []
        platform_list_train, platform_list_val, platform_list_test = [], [], []

        # loop through all experiments
        for folder in self.experiment_folders:

            # Get all NPZ files from each experiment
            folder_path = os.path.join(self.dataset_path, folder)
            X, y, tissue, platform = self._load_experiment(folder_path)

            # split data according to specified ratios
            X_train_batch, y_train_batch, X_val_batch, y_val_batch, X_test_batch, y_test_batch = \
                train_val_test_split(X_data=X, y_data=y, data_split=data_split, seed=seed)

            # construct list for each split
            tissue_list_train_batch = [tissue] * X_train_batch.shape[0]
            platform_list_train_batch = [platform] * X_train_batch.shape[0]

            tissue_list_val_batch = [tissue] * X_val_batch.shape[0]
            platform_list_val_batch = [platform] * X_val_batch.shape[0]

            tissue_list_test_batch = [tissue] * X_test_batch.shape[0]
            platform_list_test_batch = [platform] * X_test_batch.shape[0]

            # append batch to main list
            X_train.append(X_train_batch)
            X_val.append(X_val_batch)
            X_test.append(X_test_batch)

            y_train.append(y_train_batch)
            y_val.append(y_val_batch)
            y_test.append(y_test_batch)

            tissue_list_train.append(tissue_list_train_batch)
            tissue_list_val.append(tissue_list_val_batch)
            tissue_list_test.append(tissue_list_test_batch)

            platform_list_train.append(platform_list_train_batch)
            platform_list_val.append(platform_list_val_batch)
            platform_list_test.append(platform_list_test_batch)

        # make sure that all data has same shape
        first_shape = X_train[0].shape
        for i in range(1, len(X_train)):
            current_shape = X_train[i].shape
            if first_shape[1:] != current_shape[1:]:
                raise ValueError('Found mismatching dimensions between '
                                 'first NPZ and npz at position {}. '
                                 'Shapes of {}, {}'.format(i, first_shape, current_shape))

        # concatenate lists together
        X_train = np.concatenate(X_train, axis=0)
        X_val = np.concatenate(X_val, axis=0)
        X_test = np.concatenate(X_test, axis=0)

        y_train = np.concatenate(y_train, axis=0)
        y_val = np.concatenate(y_val, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        tissue_list_train = np.concatenate(tissue_list_train, axis=0)
        tissue_list_val = np.concatenate(tissue_list_val, axis=0)
        tissue_list_test = np.concatenate(tissue_list_test, axis=0)

        platform_list_train = np.concatenate(platform_list_train, axis=0)
        platform_list_val = np.concatenate(platform_list_val, axis=0)
        platform_list_test = np.concatenate(platform_list_test, axis=0)

        # create combined dicts
        train_dict = {'X': X_train, 'y': y_train, 'tissue_list': tissue_list_train,
                      'platform_list': platform_list_train}

        val_dict = {'X': X_val, 'y': y_val, 'tissue_list': tissue_list_val,
                    'platform_list': platform_list_val}

        test_dict = {'X': X_test, 'y': y_test, 'tissue_list': tissue_list_test,
                     'platform_list': platform_list_test}

        self.train_dict = train_dict
        self.val_dict = val_dict
        self.test_dict = test_dict
        self.data_split = data_split
        self.seed = seed

    def _subset_data_dict(self, data_dict, tissues, platforms):
        """Subsets a dictionary to only include from the specified tissues and platforms

        Args:
            data_dict: dictionary to subset from
            tissues: list of tissues to include
            platforms: list of platforms to include

        Returns:
            subset_dict: dictionary containing examples desired data

        Raises:
            ValueError: If no matching data for tissue/platform combination
        """
        X, y = data_dict['X'], data_dict['y']
        tissue_list, platform_list = data_dict['tissue_list'], data_dict['platform_list']

        # Identify locations with the correct categories types
        tissue_idx = np.isin(tissue_list, tissues)
        platform_idx = np.isin(platform_list, platforms)

        # get indices which meet both criteria
        combined_idx = tissue_idx * platform_idx

        # check that there is data which meets requirements
        if np.sum(combined_idx) == 0:
            raise ValueError('No matching data for specified parameters')

        X, y = X[combined_idx], y[combined_idx]
        tissue_list = np.array(tissue_list)[combined_idx]
        platform_list = np.array(platform_list)[combined_idx]

        subset_dict = {'X': X, 'y': y, 'tissue_list': list(tissue_list),
                       'platform_list': list(platform_list)}
        return subset_dict

    def _reshape_dict(self, dict, resize=False, output_shape=(512, 512), resize_target=400,
                      resize_tolerance=1.5):
        """Takes a dictionary of training data and reshapes it to appropriate size

        dict: dictionary of training data
        resize: flag to control resizing of the data.
            Valid arguments:
                    - False. No resizing
                    - by_tissue. Resizes by median cell size within each tissue type
                    - by_image. Resizes by median cell size within each image
        output_shape: output shape for image data
        resize_target: desired median cell size after resizing
        resize_tolerance: sets maximum allowable ratio between resize_target and
            median cell size before resizing occurs
        """
        X, y = dict['X'], dict['y']
        tissue_list, platform_list = np.array(dict['tissue_list']), np.array(dict['platform_list'])

        if not resize:
            # no resizing
            X_new, y_new = reshape_training_data(X_data=X, y_data=y, resize_ratio=1,
                                                 final_size=output_shape, stride_ratio=1)

            # to preserve category labels, we need to figure out how much the array grew by
            multiplier = int(X_new.shape[0] / X.shape[0])

            # then we duplicate the labels in place to match expanded array size
            tissue_list_new = [item for item in tissue_list for _ in range(multiplier)]
            platform_list_new = [item for item in platform_list for _ in range(multiplier)]

        elif resize == 'by_tissue':
            X_new, y_new, tissue_list_new, platform_list_new = [], [], [], []
            unique_tissues = np.unique(tissue_list)

            # loop through each unique tissue and resize images by tissue median
            for uid in unique_tissues:
                # subset lists to include only current tissue
                uid_idx = np.isin(tissue_list, uid)
                X_uid, y_uid = X[uid_idx], y[uid_idx]
                tissue_list_uid, platform_list_uid = tissue_list[uid_idx], platform_list[uid_idx]

                # compute appropriate resize ratio
                median_cell_size = compute_cell_size({'X': X_uid, 'y': y_uid})
                resize_ratio = median_cell_size / resize_target

                # resize the data
                X_uid_resized, y_uid_resized = reshape_training_data(X_data=X_uid, y_data=y_uid,
                                                                     resize_ratio=resize_ratio,
                                                                     final_size=output_shape,
                                                                     tolerance=resize_tolerance)

                # to preserve category labels, we need to figure out how much the array grew by
                multiplier = int(X_uid_resized.shape[0] / X_uid.shape[0])
                # then we duplicate the labels in place to match expanded array size
                tissue_list_uid = [item for item in tissue_list_uid for _ in range(multiplier)]
                platform_list_uid = [item for item in platform_list_uid for _ in range(multiplier)]

                # add each batch onto main list
                X_new.append(X_uid_resized)
                y_new.append(y_uid_resized)
                tissue_list_new.append(tissue_list_uid)
                platform_list_new.append(platform_list_uid)

            X_new = np.concatenate(X_new, axis=0)
            y_new = np.concatenate(y_new, axis=0)
            tissue_list_new = np.concatenate(tissue_list_new, axis=0)
            platform_list_new = np.concatenate(platform_list_new, axis=0)
        else:
            # resize each image individually
            X_new, y_new, tissue_list_new, platform_list_new = [], [], [], []

            # loop through each unique tissue and resize images by tissue median
            for img in range(X.shape[0]):
                # subset lists to include only current tissue
                X_batch, y_batch = X[img:(img + 1)], y[img:(img + 1)]
                tissue_list_batch = tissue_list[img:(img + 1)]
                platform_list_batch = platform_list[img:(img + 1)]

                # compute appropriate resize ratio
                median_cell_size = compute_cell_size({'X': X_batch, 'y': y_batch})
                resize_ratio = median_cell_size / resize_target

                # resize the data
                X_batch_resized, y_batch_resized = \
                    reshape_training_data(X_data=X_batch, y_data=y_batch,
                                          resize_ratio=resize_ratio, final_size=output_shape,
                                          tolerance=resize_tolerance)

                # to preserve category labels, we need to figure out how much the array grew by
                multiplier = int(X_batch_resized.shape[0] / X_batch.shape[0])

                # then we duplicate the labels in place to match expanded array size
                tissue_list_batch = [item for item in tissue_list_batch for _ in range(multiplier)]
                platform_list_batch = \
                    [item for item in platform_list_batch for _ in range(multiplier)]

                # add each batch onto main list
                X_new.append(X_batch_resized)
                y_new.append(y_batch_resized)
                tissue_list_new.append(tissue_list_batch)
                platform_list_new.append(platform_list_batch)

            X_new = np.concatenate(X_new, axis=0)
            y_new = np.concatenate(y_new, axis=0)
            tissue_list_new = np.concatenate(tissue_list_new, axis=0)
            platform_list_new = np.concatenate(platform_list_new, axis=0)

        return {'X': X_new, 'y': y_new, 'tissue_list': tissue_list_new,
                'platform_list': platform_list_new}

    def _clean_labels(self, y, relabel_hard=False, small_object_threshold=0, min_objects=0):
        """Cleans labels prior to creating final dict

        Args:
            relabel_hard: if True, relabels image with label. Otherwise,
                uses relabel_sequential.
            small_object_threshold: threshold for removing small objects
            min_objects: minimum number of objects per image

        Returns:
            cleaned_y: labels with relevant transformations applied
        """

        keep_idx = np.repeat(True, y.shape[0])
        cleaned_y = np.zeros_like(y)

        for i in range(y.shape[0]):
            if relabel_hard:
                y_new = label(y[i, :, :, 0])
            else:
                y_new, _, _ = relabel_sequential(y[i, :, :, 0])

            y_new = remove_small_objects(y_new, min_size=small_object_threshold)

            unique_objects = len(np.unique(y_new)) - 1
            if unique_objects < min_objects:
                keep_idx[i] = False

            cleaned_y[i, :, :, 0] = y_new

        cleaned_y = cleaned_y[keep_idx]

        return cleaned_y

    def build_dataset(self, tissues='all', platforms='all', output_shape=(512, 512), resize=False,
                      data_split=(0.8, 0.1, 0.1), seed=0):
        """Construct a dataset for model training and evaluation

        Args:
            tissues: which tissues to include. Must be either a list of tissue types,
                a single tissue type, or 'all'
            platforms: which platforms to include. Must be either a list of platform types,
                a single platform type, or 'all'
            output_shape: output shape for dataset
            resize: flag to control resizing the input data.
                Valid arguments:
                    - False. No resizing
                    - by_tissue. Resizes by median cell size within each tissue type
                    - by_image. REsizes by median cell size within each image
            data_split: tuple specifying the fraction of the dataset for train/val/test
            seed: seed for reproducible splitting of dataset

        Returns:
            list of dicts containing the split dataset

        Raises:
            ValueError: If invalid tissues argument specified
            ValueError: If invalid platforms argument specified
        """
        if self.all_tissues == []:
            self._identify_tissue_and_platform_types()

        # validate inputs
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
                'tissues should be "all", one of {}, or a list '
                'of acceptable tissue types'.format(self.all_tissues))

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
                'platforms should be "all", one of {}, or a list of acceptable '
                'platform types'.format(self.all_platforms))

        valid_resize = [False, 'by_tissue', 'by_image']
        if resize not in valid_resize:
            raise ValueError('resize must be one of {}'.format(valid_resize))

        # if any of the split parameters are different we need to reload the dataset
        if self.seed != seed or self.data_split != data_split:
            self._load_all_experiments(data_split=data_split, seed=seed)

        dicts = [self.train_dict, self.val_dict, self.test_dict]

        # process each dict
        for idx, current_dict in enumerate(dicts):
            # subset dict to include only relevant tissues and platforms
            current_dict = self._subset_data_dict(data_dict=current_dict, tissues=tissues,
                                                  platforms=platforms)

            # if necessary, reshape and resize data to be of correct output size
            if current_dict['X'].shape[1:3] != output_shape or resize is not False:
                current_dict = self._reshape_dict(dict=current_dict, resize=resize,
                                                  output_shape=output_shape)
            # clean labels
            cleaned_labels = self._clean_labels(current_dict['y'])
            current_dict['y'] = cleaned_labels

            dicts[idx] = current_dict

        return dicts
