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

import math

import numpy as np

from skimage.measure import regionprops_table
from sklearn.model_selection import train_test_split


from deepcell_toolbox.utils import resize, tile_image


def compute_cell_size(npz_file, method='median', by_image=True):
    """Computes the typical cell size from a stack of labeled data

    Args:
        npz_file: Paired X and y data
        method: one of (mean, median) used to compute the cell size
        by_image: if true, cell size is reported for each image in npz_file. Otherwise,
            the cell size across the entire npz is returned

    Returns:
        average_sizes: list of typical cell size in NPZ. If no cells, returns None

    Raises: ValueError if invalid method supplied
    Raises: ValueError if data does have len(shape) of 4
    """

    valid_methods = {'median', 'mean'}
    method = method.lower()
    if method not in valid_methods:
        raise ValueError('Invalid method supplied: got {}, '
                         'method must be one of {}'.format(method, valid_methods))

    # initialize variables
    cell_sizes = []
    labels = npz_file['y']

    if len(labels.shape) != 4:
        raise ValueError('Labeled data must be 4D')

    for i in range(labels.shape[0]):
        current_label = labels[i, :, :, 0]

        # check to make sure array contains cells
        if len(np.unique(current_label)) > 1:
            area = regionprops_table(current_label.astype('int'), properties=['area'])['area']
            cell_sizes.append(area)

    # if all images were empty, return NA
    if cell_sizes == []:
        return None

    # compute for each list corresponding to each image
    if by_image:
        average_cell_sizes = []
        for size_list in cell_sizes:
            if method == 'mean':
                average_cell_sizes.append(np.mean(size_list))
            elif method == 'median':
                average_cell_sizes.append(np.median(size_list))

    # compute across all lists from all images
    else:
        all_cell_sizes = np.concatenate(cell_sizes)
        if method == 'mean':
            average_cell_sizes = np.mean(all_cell_sizes)
        elif method == 'median':
            average_cell_sizes = np.median(all_cell_sizes)
        else:
            raise ValueError('Invalid method supplied')

    return average_cell_sizes


def reshape_training_data(X_data, y_data, resize_ratio, final_size, stride_ratio=1, tolerance=1.5):
    """Takes a stack of X and y data and reshapes and crops them to match output dimensions

    Args:
        X_data: 4D numpy array of image data
        y_data: 4D numpy array of labeled data
        resize_ratio: resize ratio for the images
        final_size: the desired shape of the output image
        stride_ratio: amount of overlap between crops (1 is no overlap, 0.5 is half crop size)
        tolerance: ratio that determines when resizing occurs

    Returns:
        reshaped_X, reshaped_y: resized and cropped version of input images
    """

    # resize if needed
    if resize_ratio > tolerance or resize_ratio < (1 / tolerance):
        new_shape = (int(X_data.shape[1] * resize_ratio),
                     int(X_data.shape[2] * resize_ratio))

        X_data = resize(data=X_data, shape=new_shape)
        y_data = resize(data=y_data, shape=new_shape, labeled_image=True)

    # crop if needed
    if X_data.shape[1:3] != final_size:
        # pad image so that crops divide evenly
        X_data = pad_image_stack(images=X_data, crop_size=final_size)
        y_data = pad_image_stack(images=y_data, crop_size=final_size)

        # create x and y crops
        X_data, _ = tile_image(image=X_data, model_input_shape=final_size,
                               stride_ratio=stride_ratio)
        y_data, _ = tile_image(image=y_data, model_input_shape=final_size,
                               stride_ratio=stride_ratio)
    return X_data, y_data


def pad_image_stack(images, crop_size):
    """Pads an an array of images so that it is divisible by the specified crop_size

    Args:
        images: array of images to be cropped
        crop_size: tuple specifying crop size

    Returns:
        np.array: padded image stack
    """

    row_len, col_len = images.shape[1:3]
    row_crop, col_crop = crop_size
    row_num = math.ceil(row_len / crop_size[0])
    col_num = math.ceil(col_len / crop_size[1])

    new_row_len = row_num * row_crop
    new_col_len = col_num * col_crop

    if new_row_len == row_len and new_col_len == col_len:
        # don't need to pad
        return images
    else:
        new_images = np.zeros((images.shape[0], new_row_len, new_col_len, images.shape[3]))
        new_images[:, :row_len, :col_len, :] = images
        return new_images


def train_val_test_split(X_data, y_data, data_split=(0.8, 0.1, 0.1), seed=None):
    """Randomly splits supplied data into specified sizes for model assessment

    Args:
        X_data: array of X data
        y_data: array of y_data
        data_split: tuple specifying the fraction of the dataset for train/val/test
        seed: random seed for reproducible splits

    Returns:
        list of X and y data split appropriately

    Raises:
        ValueError: if ratios do not sum to 1
        ValueError: if any of the splits are 0
        ValueError: If length of X and y data is not equal
    """

    total = np.round(np.sum(data_split), decimals=2)
    if total != 1:
        raise ValueError('Data splits must sum to 1, supplied splits sum to {}'.format(total))

    if 0 in data_split:
        raise ValueError('All splits must be non-zero')

    if X_data.shape[0] != y_data.shape[0]:
        raise ValueError('Supplied X and y data do not have the same '
                         'length over batches dimension. '
                         'X.shape: {}, y.shape: {}'.format(X_data.shape, y_data.shape))

    train_ratio, val_ratio, test_ratio = data_split

    # compute fraction not in train
    remainder_size = np.round(1 - train_ratio, decimals=2)

    # split dataset into train and remainder
    X_train, X_remainder, y_train, y_remainder = train_test_split(X_data, y_data,
                                                                  test_size=remainder_size,
                                                                  random_state=seed)

    # compute fraction of remainder that is test
    test_size = np.round(test_ratio / (val_ratio + test_ratio), decimals=2)

    # split remainder into val and test
    X_val, X_test, y_val, y_test = train_test_split(X_remainder, y_remainder,
                                                    test_size=test_size,
                                                    random_state=seed)

    return X_train, y_train, X_val, y_val, X_test, y_test
