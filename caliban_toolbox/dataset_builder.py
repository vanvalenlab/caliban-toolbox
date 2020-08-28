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
import warnings

import numpy as np

from skimage.measure import label
from skimage.morphology import remove_small_objects

from caliban_toolbox.utils.misc_utils import list_npzs_folder, list_folders
from caliban_toolbox.build import train_val_test_split, reshape_training_data, compute_cell_size


class DatasetBuilder(object):
    """Class to build a dataset from annotated data

    Args:
        dataset_path: path to dataset. Within the dataset, each unique experiment
            has its own folder with a dedicated metadata file
    """
    def __init__(self, dataset_path):

        self._validate_dataset(dataset_path)

        experiment_folders = list_folders(dataset_path)
        self.dataset_path = dataset_path
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

    def _validate_dataset(self, dataset_path):
        """Check to make sure that supplied dataset is formatted appropriately

        Args:
            dataset_path: path to dataset

        Raises:
            ValueError: If dataset_path doesn't exist
            ValueError: If dataset_path doesn't contain any folders
            ValueError: If dataset_path has any folders without an NPZ file
            ValueError: If dataset_path has any folders without a metadata file
        """

        if not os.path.isdir(dataset_path):
            raise ValueError('Invalid dataset_path, must be a directory')
        experiment_folders = list_folders(dataset_path)

        if experiment_folders == []:
            raise ValueError('dataset_path must include at least one folder')

        for folder in experiment_folders:
            if not os.path.exists(os.path.join(dataset_path, folder, 'metadata.json')):
                raise ValueError('No metadata file found in {}'.format(folder))
            npz_files = list_npzs_folder(os.path.join(dataset_path, folder))

            if len(npz_files) == 0:
                raise ValueError('No NPZ files found in {}'.format(folder))

    def _get_metadata(self, experiment_folder):
        """Get the metadata associated with a specific experiment

        Args:
            experiment_folder: folder to get metadata from

        Returns:
            dictionary containing relevant metadata"""

        metadata_file = os.path.join(self.dataset_path, experiment_folder, 'metadata.json')
        with open(metadata_file) as f:
            metadata = json.load(f)

        return metadata

    def _identify_tissue_and_platform_types(self):
        """Identify all of the unique tissues and platforms in the dataset"""

        tissues = []
        platforms = []
        for folder in self.experiment_folders:
            metadata = self._get_metadata(experiment_folder=folder)

            tissues.append(metadata['tissue'])
            platforms.append(metadata['platform'])

        self.all_tissues.extend(tissues)
        self.all_platforms.extend(platforms)

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
        metadata = self._get_metadata(experiment_folder=experiment_path)

        # combine all NPZ files together
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        if isinstance(y.dtype, np.floating):
            warnings.warn('Converting float labels to integers')
            y = y.astype('int64')

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
            X_train.append(X_train_batch)
            y_train.append(y_train_batch)
            tissue_list_train.append(tissue_list_train_batch)
            platform_list_train.append(platform_list_train_batch)

            if X_val_batch is not None:
                tissue_list_val_batch = [tissue] * X_val_batch.shape[0]
                platform_list_val_batch = [platform] * X_val_batch.shape[0]
                X_val.append(X_val_batch)
                y_val.append(y_val_batch)
                tissue_list_val.append(tissue_list_val_batch)
                platform_list_val.append(platform_list_val_batch)

            if X_test_batch is not None:
                tissue_list_test_batch = [tissue] * X_test_batch.shape[0]
                platform_list_test_batch = [platform] * X_test_batch.shape[0]
                X_test.append(X_test_batch)
                y_test.append(y_test_batch)
                tissue_list_test.append(tissue_list_test_batch)
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

    def _reshape_dict(self, data_dict, resize=False, output_shape=(512, 512), resize_target=400,
                      resize_tolerance=1.5):
        """Takes a dictionary of training data and reshapes it to appropriate size

        data_dict: dictionary of training data
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
        X, y = data_dict['X'], data_dict['y']
        tissue_list = np.array(data_dict['tissue_list'])
        platform_list = np.array(data_dict['platform_list'])

        if not resize:
            # no resizing
            X_new, y_new = reshape_training_data(X_data=X, y_data=y, resize_ratio=1,
                                                 final_size=output_shape, stride_ratio=1)

            # to preserve category labels, we need to figure out how much the array grew by
            multiplier = int(X_new.shape[0] / X.shape[0])

            # then we duplicate the labels in place to match expanded array size
            tissue_list_new = [item for item in tissue_list for _ in range(multiplier)]
            platform_list_new = [item for item in platform_list for _ in range(multiplier)]

        else:
            X_new, y_new, tissue_list_new, platform_list_new = [], [], [], []

            if resize == 'by_tissue':
                batch_ids = np.unique(tissue_list)
            else:
                batch_ids = np.arange(0, X.shape[0])

            # loop over each batch
            for batch_id in batch_ids:

                # get tissue types that match current tissue type
                if isinstance(batch_id, str):
                    batch_idx = np.isin(tissue_list, batch_id)

                # get boolean index for current image
                else:
                    batch_idx = np.arange(X.shape[0]) == batch_id

                X_batch, y_batch = X[batch_idx], y[batch_idx]
                tissue_list_batch = tissue_list[batch_idx]
                platform_list_batch = platform_list[batch_idx]

                # compute appropriate resize ratio
                median_cell_size = compute_cell_size({'X': X_batch, 'y': y_batch}, by_image=False)

                # check for empty images
                if median_cell_size is not None:
                    resize_ratio = np.sqrt(resize_target / median_cell_size)
                else:
                    resize_ratio = 1

                # resize the data
                X_batch_resized, y_batch_resized = reshape_training_data(
                    X_data=X_batch, y_data=y_batch,
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

    def _clean_labels(self, data_dict, relabel=False, small_object_threshold=0,
                      min_objects=0):
        """Cleans labels prior to creating final dict

        Args:
            data_dict: dictionary of training data
            relabel: if True, relabels the image with new labels
            small_object_threshold: threshold for removing small objects
            min_objects: minimum number of objects per image

        Returns:
            cleaned_dict: dictionary with cleaned labels
        """
        X, y = data_dict['X'], data_dict['y']
        tissue_list = np.array(data_dict['tissue_list'])
        platform_list = np.array(data_dict['platform_list'])
        keep_idx = np.repeat(True, y.shape[0])
        cleaned_y = np.zeros_like(y)

        # TODO: remove one data QC happens in main toolbox pipeline
        for i in range(y.shape[0]):
            y_current = y[i, ..., 0]
            if relabel:
                y_current = label(y_current)

            y_current = remove_small_objects(y_current, min_size=small_object_threshold)

            unique_objects = len(np.unique(y_current)) - 1
            if unique_objects < min_objects:
                keep_idx[i] = False

            cleaned_y[i, ..., 0] = y_current

        # subset all dict members to include only relevant images
        cleaned_y = cleaned_y[keep_idx]
        cleaned_X = X[keep_idx]
        cleaned_tissue = tissue_list[keep_idx]
        cleaned_platform = platform_list[keep_idx]

        cleaned_dict = {'X': cleaned_X, 'y': cleaned_y, 'tissue_list': list(cleaned_tissue),
                        'platform_list': list(cleaned_platform)}

        return cleaned_dict

    def _validate_categories(self, category_list, supplied_categories):
        """Check that an appropriate subset of a list of categories was supplied

        Args:
            category_list: list of all categories
            supplied_categories: specified categories provided by user. Must be either
                - a list containing the desired category members
                - a string of a single category name
                - a string of 'all', in which case all will be used

        Returns:
            list: a properly formatted sub_category list

        Raises:
            ValueError: if invalid supplied_categories argument
            """
        if isinstance(supplied_categories, list):
            for cat in supplied_categories:
                if cat not in category_list:
                    raise ValueError('{} is not one of {}'.format(cat, category_list))
            return supplied_categories
        elif supplied_categories == 'all':
            return category_list
        elif supplied_categories in category_list:
            return [supplied_categories]
        else:
            raise ValueError(
                'Specified categories should be "all", one of {}, or a list '
                'of acceptable tissue types'.format(category_list))

    def build_dataset(self, tissues='all', platforms='all', output_shape=(512, 512), resize=False,
                      data_split=(0.8, 0.1, 0.1), seed=0, **kwargs):
        """Construct a dataset for model training and evaluation

        Args:
            tissues: which tissues to include. Must be either a list of tissue types,
                a single tissue type, or 'all'
            platforms: which platforms to include. Must be either a list of platform types,
                a single platform type, or 'all'
            output_shape: output shape for dataset. Either a single tuple, in which case
                train/va/test will all have same size, or a list of three tuples
            resize: flag to control resizing the input data.
                Valid arguments:
                    - False. No resizing
                    - by_tissue. Resizes by median cell size within each tissue type
                    - by_image. REsizes by median cell size within each image
            data_split: tuple specifying the fraction of the dataset for train/val/test
            seed: seed for reproducible splitting of dataset
            **kwargs: other arguments to be passed to helper functions

        Returns:
            list of dicts containing the split dataset

        Raises:
            ValueError: If invalid resize parameter supplied
            ValueError: If invalid output_shape parameter supplied
        """
        if self.all_tissues == []:
            self._identify_tissue_and_platform_types()

        # validate inputs
        tissues = self._validate_categories(category_list=self.all_tissues,
                                            supplied_categories=tissues)
        platforms = self._validate_categories(category_list=self.all_platforms,
                                              supplied_categories=platforms)

        valid_resize = [False, 'by_tissue', 'by_image']
        if resize not in valid_resize:
            raise ValueError('resize must be one of {}'.format(valid_resize))

        if not isinstance(output_shape, (list, tuple)):
            raise ValueError('output_shape must be either a list of tuples or a tuple')

        # convert from single tuple to list of tuples for each split
        if isinstance(output_shape, tuple):
            output_shape = [output_shape, output_shape, output_shape]

        for tup in output_shape:
            if len(tup) != 2:
                raise ValueError('Each output_shape must be len(2) tuple, got {}'.format(tup))

        # if any of the split parameters are different we need to reload the dataset
        if self.seed != seed or self.data_split != data_split:
            self._load_all_experiments(data_split=data_split, seed=seed)

        dicts = [self.train_dict, self.val_dict, self.test_dict]

        # process each dict
        for idx, current_dict in enumerate(dicts):
            # subset dict to include only relevant tissues and platforms
            current_dict = self._subset_data_dict(data_dict=current_dict, tissues=tissues,
                                                  platforms=platforms)
            current_shape = output_shape[idx]

            # if necessary, reshape and resize data to be of correct output size
            if current_dict['X'].shape[1:3] != current_shape or resize is not False:
                resize_target = kwargs.get('resize_target', 400)
                resize_tolerance = kwargs.get('resize_tolerance', 1.5)
                current_dict = self._reshape_dict(data_dict=current_dict, resize=resize,
                                                  output_shape=current_shape,
                                                  resize_target=resize_target,
                                                  resize_tolerance=resize_tolerance)
            # clean labels
            relabel = kwargs.get('relabel', False)
            print('relabel arg is {}'.format(relabel))
            small_object_threshold = kwargs.get('small_object_threshold', 0)
            min_objects = kwargs.get('min_objects', 0)
            current_dict = self._clean_labels(data_dict=current_dict, relabel=relabel,
                                              small_object_threshold=small_object_threshold,
                                              min_objects=min_objects)
            print("index is {}, unique is {}".format(idx, np.unique(current_dict['y'])))
            dicts[idx] = current_dict

        return dicts

    def summarize_dataset(self):
        """Computes summary statistics for the images in the dataset

        Returns:
            dict of cell counts and image counts by tissue
            dict of cell counts and image counts by platform
        """
        all_y = np.concatenate((self.train_dict['y'],
                                self.val_dict['y'],
                                self.test_dict['y']),
                               axis=0)
        all_tissue = np.concatenate((self.train_dict['tissue_list'],
                                     self.val_dict['tissue_list'],
                                     self.test_dict['tissue_list']),
                                    axis=0)

        all_platform = np.concatenate((self.train_dict['platform_list'],
                                       self.val_dict['platform_list'],
                                       self.test_dict['platform_list']),
                                      axis=0)
        all_counts = np.zeros(all_y.shape[0])
        for i in range(all_y.shape[0]):
            unique_counts = len(np.unique(all_y[i, ..., 0])) - 1
            all_counts[i] = unique_counts

        tissue_dict = {}
        for tissue in np.unique(all_tissue):
            tissue_idx = np.isin(all_tissue, tissue)
            tissue_counts = np.sum(all_counts[tissue_idx])
            tissue_unique = np.sum(tissue_idx)
            tissue_dict[tissue] = {'cell_num': tissue_counts,
                                   'image_num': tissue_unique}

        platform_dict = {}
        for platform in np.unique(all_platform):
            platform_idx = np.isin(all_platform, platform)
            platform_counts = np.sum(all_counts[platform_idx])
            platform_unique = np.sum(platform_idx)
            platform_dict[platform] = {'cell_num': platform_counts,
                                       'image_num': platform_unique}

        return tissue_dict, platform_dict
