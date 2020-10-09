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
import numpy as np


class DatasetSplitter(object):
    def __init__(self, seed=0):
        """Class to split a dataset into sequentially increasing tranches for model training

        Args:
            seed: random seed for splitting
        """

        self.seed = seed

    def _validate_dict(self, train_dict):
        if 'X' not in train_dict or 'y' not in train_dict:
            raise ValueError('X and y must be keys in the training dictionary')

    def _validate_split_counts(self, split_counts):
        """Ensures that split_counts are properly formatted"""

        split_counts.sort()
        if split_counts[0] <= 0:
            raise ValueError('Smallest split_count must be greater than 0, '
                             'got {}'.format(split_counts[0]))

        ids, counts = np.unique(split_counts, return_counts=True)
        if np.any(counts != 1):
            raise ValueError('Duplicate split_counts are not allowed, '
                             'each split must be unique')
        dtypes = [isinstance(x, int) for x in split_counts]
        if not np.all(dtypes):
            raise ValueError('All split_counts must be integers')

        return split_counts

    def _validate_split_proportions(self, split_proportions):
        """Ensures that split_proportions are properly formatted"""

        split_proportions.sort()
        if split_proportions[0] <= 0:
            raise ValueError('Smallest split_proportion must be non-zero, '
                             'got {}'.format(split_proportions[0]))
        if split_proportions[-1] > 1:
            raise ValueError('Largest split_proportion cannot be greater than 1, '
                             'got {}'.format(split_proportions[-1]))
        ids, counts = np.unique(split_proportions, return_counts=True)
        if np.any(counts != 1):
            raise ValueError('Duplicate splits are not allowed, each split must be uniqe')

        return split_proportions

    def _duplicate_indices(self, indices, min_size):
        """Duplicates supplied indices to that there are min_size number

        Args:
            indices: array specifying indices of images to be included
            min_size: minimum number of images in split

        Returns:
            array: duplicate indices
        """

        multiplier = int(np.ceil(min_size / len(indices)))
        new_indices = np.tile(indices, multiplier)
        new_indices = new_indices[:min_size]

        return new_indices

    def split(self, input_dict, split_counts=None, split_proportions=None, min_size=1):
        """Split training dict

        Args:
            input_dict: dictionary containing paired X and y data
            split_counts: list with number of images from total dataset in each split
            split_proportions: list with fraction of total dataset in each split
            min_size: minimum number of images for each split. If supplied split size leads to a
                split with fewer than min_size, duplicates included images up to specified count

        Returns:
            dict: dict of dicts containing each split

        Raises:
            ValueError: If split_counts and split_proportions are both None
        """
        self._validate_dict(input_dict)

        X = input_dict['X']
        y = input_dict['y']
        N_batches = X.shape[0]

        if split_counts is None and split_proportions is None:
            raise ValueError('Either split_counts or split_proportions must be supplied')

        if split_counts is not None:
            if split_proportions is not None:
                raise ValueError('Either split_counts or split_proportions must be supplied,'
                                 'not both')
            # get counts per split and key used to store the split
            split_counts = self._validate_split_counts(split_counts=split_counts)
            split_keys = split_counts

        if split_proportions is not None:
            split_props = self._validate_split_proportions(split_proportions=split_proportions)

            # get counts per split and key used to store the split
            split_counts = [max(int(N_batches * split_prop), 1) for split_prop in split_props]
            split_keys = split_props

        # randomize index so that we can take sequentially larger splits
        index = np.arange(N_batches)

        # randomize index so that we can take sequentially larger splits
        permuted_index = np.random.RandomState(seed=self.seed).permutation(index)

        split_dict = {}
        for idx, val in enumerate(split_counts):
            split_idx = permuted_index[0:val]

            # duplicate indices up to minimum batch size if necessary
            if len(split_idx) < min_size:
                split_idx = self._duplicate_indices(indices=split_idx, min_size=min_size)

            new_train_dict = {}
            new_train_dict['X'] = X[split_idx]
            new_train_dict['y'] = y[split_idx]
            split_dict[str(split_keys[idx])] = new_train_dict

        return split_dict
