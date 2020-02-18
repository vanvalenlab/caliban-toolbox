# Copyright 2016-2019 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-toolbox/LICENSE
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
'''
Make npz formatted training data at end of crowd-sourced annotation pipelines
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import random

from imageio import imread
import numpy as np
from deepcell_toolbox.utils.io_utils import get_img_names, get_image


def reshape_matrix(X, y, reshape_size=256, data_format="channels_last"):
    '''
    Reshape matrix of dimension 4 to have x and y of size reshape_size.
    Adds overlapping slices to batches.
    E.g. reshape_size of 256 yields (1, 1024, 1024, 1) -> (16, 256, 256, 1)

    Args:
        X: raw 4D image tensor
        y: label mask of 4D image data
        reshape_size: size of the square output tensor
        data_format: channels_first or channels_last

    Returns:
        reshaped `X` and `y` tensors in shape (`reshape_size`, `reshape_size`)
    '''
    is_channels_first = data_format == 'channels_first'
    if X.ndim != 4:
        raise ValueError('reshape_matrix expects X dim to be 4, got', X.ndim)
    elif y.ndim != 4:
        raise ValueError('reshape_matrix expects y dim to be 4, got', y.ndim)

    image_size_x, _ = X.shape[2:] if is_channels_first else X.shape[1:3]
    rep_number = np.int(np.ceil(np.float(image_size_x) / np.float(reshape_size)))
    new_batch_size = X.shape[0] * (rep_number) ** 2

    if is_channels_first:
        new_X_shape = (new_batch_size, X.shape[1], reshape_size, reshape_size)
        new_y_shape = (new_batch_size, y.shape[1], reshape_size, reshape_size)
    else:
        new_X_shape = (new_batch_size, reshape_size, reshape_size, X.shape[3])
        new_y_shape = (new_batch_size, reshape_size, reshape_size, y.shape[3])

    new_X = np.zeros(new_X_shape, dtype=K.floatx())
    new_y = np.zeros(new_y_shape, dtype='int32')

    counter = 0
    for b in range(X.shape[0]):
        for i in range(rep_number):
            for j in range(rep_number):
                if i != rep_number - 1:
                    x_start, x_end = i * reshape_size, (i + 1) * reshape_size
                else:
                    x_start, x_end = -reshape_size, X.shape[2 if is_channels_first else 1]

                if j != rep_number - 1:
                    y_start, y_end = j * reshape_size, (j + 1) * reshape_size
                else:
                    y_start, y_end = -reshape_size, y.shape[3 if is_channels_first else 2]

                if is_channels_first:
                    new_X[counter] = X[b, :, x_start:x_end, y_start:y_end]
                    new_y[counter] = y[b, :, x_start:x_end, y_start:y_end]
                else:
                    new_X[counter] = X[b, x_start:x_end, y_start:y_end, :]
                    new_y[counter] = y[b, x_start:x_end, y_start:y_end, :]

                counter += 1

    print('Reshaped feature data from {} to {}'.format(y.shape, new_y.shape))
    print('Reshaped training data from {} to {}'.format(X.shape, new_X.shape))
    return new_X, new_y


def post_annotation_load_training_images_3d(training_dir,
                            training_folders,
                            channel_names,
                            image_size,
                            num_frames,
                            data_format="channels_last"):
    '''
    Load each image in the training_folders into a numpy array.

    Args:
        training_dir: full path to parent directory that contains subfolders for
            different movies
        training_folders: list of folders where each folder contains subfolders
            of channels and features
        channel_names: list of folders where each folder contains a different channel
            of raw data to load into npz
        image_size: size of each image as tuple (x, y)
        num_frames: number of frames of movie to load into npz; if None, will
            default to the number of images in annotation folder
        data_format: channels_first or channels_last


    Returns:
        5D tensor of image data
    '''
    
    is_channels_first = data_format == 'channels_first'
    # Unpack size tuples
    image_size_x, image_size_y = image_size

    num_batches = len(training_folders)

    if num_frames == None:
        num_frames = len(get_img_names(os.path.join(training_dir, tranining_folders[0], channel_names[0])))

    # Initialize training data array
    if is_channels_first:
        X_shape = (num_batches, num_frames, len(channel_names), image_size_x, image_size_y)
    else:
        X_shape = (num_batches, num_frames, image_size_x, image_size_y, len(channel_names))

    X = np.zeros(X_shape, dtype=K.floatx())

    for b, movie_folder in enumerate(training_folders):
        for c, raw_folder in enumerate(channel_names):
    
            raw_list = get_img_names(os.path.join(training_dir, movie_folder, raw_folder))
            for f, frame in enumerate(raw_list):
                image_file = os.path.join(training_dir, movie_folder, raw_folder, frame)
                image_data = np.asarray(get_image(image_file), dtype=K.floatx())

                if is_channels_first:
                    X[b, f, c, :, :] = image_data
                else:
                    X[b, f, :, :, c] = image_data
    
    return X


def post_annotation_load_annotated_images_3d(training_dir,
                             training_folders,
                             annotation_folders,
                             image_size,
                             num_frames,
                             data_format="channels_last"):
    '''
    Load each annotated image in the training_folders into a numpy array.

    Args:
        training_dir: full path to parent directory that contains subfolders for
            different movies
        training_folders: list of folders where each folder contains subfolders
            of channels and features
        annotation_folders: list of folders where each folder contains a different
            annotation feature to load into npz
        image_size: size of each image as tuple (x, y)
        num_frames: number of frames of movie to load into npz; if None, will
            default to the number of images in annotation folder
        data_format: channels_first or channels_last


    Returns:
        5D tensor of label masks
    '''
    
    is_channels_first = data_format == 'channels_first'
    # Unpack size tuple
    image_size_x, image_size_y = image_size

    # wrapping single annotation name in list for consistency
    if not isinstance(annotation_folders, list):
        annotation_folders = [annotation_folders]
    
    num_batches = len(training_folders)
    
    if num_frames == None:
        num_frames = len(get_img_names(os.path.join(training_dir, training_folders[0], channel_names[0])))        

    # Initialize feature mask array
    if is_channels_first:
        y_shape = (num_batches, num_frames, len(annotation_folders), image_size_x, image_size_y)
    else:
        y_shape = (num_batches, num_frames, image_size_x, image_size_y, len(annotation_folders))

    y = np.zeros(y_shape, dtype='int32')

    for b, movie_folder in enumerate(training_folders):
        for l, annotation_folder in enumerate(annotation_folders):
        
            annotation_list = get_img_names(os.path.join(training_dir, movie_folder, annotation_folder))
            for f, frame in enumerate(annotation_list):
                image_data = get_image(os.path.join(training_dir, movie_folder, annotation_folder, frame))
            
                if is_channels_first:
                    y[b, f, l, :, :] = image_data
                else:
                    y[b, f, :, :, l] = image_data    


    return y


def post_annotation_make_training_data_3d(training_dir,
                          training_folders,
                          file_name_save,
                          channel_names,
                          annotation_folders,
                          num_frames = None,
                          reshape_size=None):
    '''
    Read all images in training folders and save as npz file.

    Args:
        training_dir: full path to parent directory that contains subfolders for
            different movies
        training_folders: list of folders where each folder contains subfolders
            of channels and features
        file_name_save: full path and file name for .npz file to save training data in
        channel_names: list of folders where each folder contains a different channel
            of raw data to load into npz
        annotation_folders: list of folders where each folder contains a different
            annotation feature to load into npz
        num_frames: number of frames of movie to load into npz; if None, will
            default to the number of images in annotation folder
        reshape_size: if provided, will reshape images to the given size (both x and
            y dimensions will be reshape_size)
        
    Returns:
        None
    '''
    # Load one file to get image sizes (assumes all images same size)
    test_img_dir  = os.path.join(training_dir, random.choice(training_folders), random.choice(channel_names))
    test_img_path = os.path.join(test_img_dir, random.choice(get_img_names(test_img_dir)))
    test_img = imread(test_img_path)
    
    image_size = test_img.shape

    X = post_annotation_load_training_images_3d(training_dir = training_dir,
                                training_folders = training_folders,
                                channel_names = channel_names,
                                image_size=image_size,
                                num_frames = num_frames)

    y = post_annotation_load_annotated_images_3d(training_dir = training_dir,
                                 training_folders = training_folders,
                                 annotation_folders = annotation_folders,
                                 image_size=image_size,
                                 num_frames = num_frames)

    if reshape_size is not None:
        X, y = reshape_matrix(X, y, reshape_size=reshape_size)

    # Save training data in npz format
    np.savez(file_name_save, X=X, y=y)
    
    return None

def post_annotation_load_training_images_2d(training_dir,
                            channel_names,
                            image_size,
                            data_format="channels_last"):
    '''
    Load each image in the training_dir into a numpy array.

    Args:
        training_dir: full path to parent directory that contains subfolders for
            channels
        channel_names: list of folders where each folder contains a different channel
            of raw data to load into npz
        image_size: size of each image as tuple (x, y)
        data_format: channels_first or channels_last

    Returns:
        4D tensor of image data
    '''
    
    is_channels_first = data_format == 'channels_first'
    # Unpack size tuples
    image_size_x, image_size_y = image_size

    num_batches = len(get_img_names(os.path.join(training_dir, channel_names[0])))

    # Initialize training data array
    if is_channels_first:
        X_shape = (num_batches, len(channel_names), image_size_x, image_size_y)
    else:
        X_shape = (num_batches, image_size_x, image_size_y, len(channel_names))

    X = np.zeros(X_shape, dtype=K.floatx())

    
    for c, raw_folder in enumerate(channel_names):
    
        raw_list = get_img_names(os.path.join(training_dir, raw_folder))
        for b, img in enumerate(raw_list):
            image_file = os.path.join(training_dir, raw_folder, img)
            image_data = np.asarray(get_image(image_file), dtype=K.floatx())

            if is_channels_first:
                X[b, c, :, :] = image_data
            else:
                X[b, :, :, c] = image_data

    return X


def post_annotation_load_annotated_images_2d(training_dir,
                             annotation_folders,
                             image_size,
                             data_format="channels_last"):
    '''
    Load each annotated image in the training_direcs into a numpy array.

    Args:
        training_dir: full path to parent directory that contains subfolders for
            annotations
        annotation_folders: list of folders where each folder contains a different
            annotation feature to load into npz
        image_size: size of each image as tuple (x, y)
        data_format: channels_first or channels_last

    Returns:
        4D tensor of label masks
    '''
    
    is_channels_first = data_format == 'channels_first'
    # Unpack size tuple
    image_size_x, image_size_y = image_size

    # wrapping single annotation name in list for consistency
    if not isinstance(annotation_folders, list):
        annotation_folders = [annotation_folders]
    
    num_batches = len(get_img_names(os.path.join(training_dir, annotation_folders[0])))

    # Initialize feature mask array
    if is_channels_first:
        y_shape = (num_batches, len(annotation_folders), image_size_x, image_size_y)
    else:
        y_shape = (num_batches, image_size_x, image_size_y, len(annotation_folders))

    y = np.zeros(y_shape, dtype='int32')

    for l, annotation_folder in enumerate(annotation_folders):
        
        annotation_list = get_img_names(os.path.join(training_dir, annotation_folder))
        for b, img in enumerate(annotation_list):
            image_data = get_image(os.path.join(training_dir, annotation_folder, img))
            
            if is_channels_first:
                y[b, l, :, :] = image_data
            else:
                y[b, :, :, l] = image_data    


    return y


def post_annotation_make_training_data_2d(training_dir,
                          file_name_save,
                          channel_names,
                          annotation_folders=['annotated'],
                          reshape_size=None):
    '''
    Read all images in training directory and save as npz file.

    Args:
        training_dir: full path to parent directory that contains folders for channels
            and features
        file_name_save: full path and file name for .npz file to save training data in
        channel_names: list of folders where each folder contains a different channel
            of raw data to load into npz
        annotation_folders: list of folders where each folder contains a different
            annotation feature to load into npz
        reshape_size: if provided, will reshape images to the given size (both x and
            y dimensions will be reshape_size)
        
    Returns:
        None
    '''
    
    # Load one file to get image sizes (assumes all images same size)
    test_img_dir  = os.path.join(training_dir, random.choice(channel_names))
    test_img_path = os.path.join(test_img_dir, random.choice(get_img_names(test_img_dir)))
    test_img = imread(test_img_path)
    
    image_size = test_img.shape

    X = post_annotation_load_training_images_2d(training_dir,
                                channel_names,
                                image_size=image_size)

    y = post_annotation_load_annotated_images_2d(training_dir,
                                 annotation_folders,
                                 image_size=image_size)

    if reshape_size is not None:
        X, y = reshape_matrix(X, y, reshape_size=reshape_size)

    # Save training data in npz format
    np.savez(file_name_save, X=X, y=y)
    
    return None
    
def post_annotation_make_training_data(training_dir,
                                       file_name_save,
                                       channel_names,
                                       annotation_folders,
                                       reshape_size,
                                       dimensionality,
                                       **kwargs):
    '''
    Wrapper function for other make_training_data functions (2D, 3D)
    Calls one of the above functions based on the dimensionality of the data.
    '''
    
    #validate arguments
    if not isinstance(dimensionality, (int, float)):
        raise ValueError('Data dimensionality should be an integer value, typically 2 or 3. '
                         'Recieved {}'.format(type(dimensionality).__name__))

    if not isinstance(channel_names, (list,)):
        raise ValueError('channel_names should be a list of strings (e.g. [\'DAPI\']). '
                         'Found {}'.format(type(channel_names).__name__))

                         
    dimensionality = int(dimensionality)

    if dimensionality == 2:
        post_annotation_make_training_data_2d(training_dir = training_dir,
                              file_name_save = file_name_save,
                              channel_names = channel_names,
                              annotation_folders = annotation_folders,
                              reshape_size=None)

    elif dimensionality == 3:
        post_annotation_make_training_data_3d(training_dir = training_dir,
                                              training_folders = kwargs.get('training_folders'),
                                              file_name_save = file_name_save,
                                              channel_names = channel_names,
                                              annotation_folders = annotation_folders,
                                              num_frames = kwargs.get('num_frames', None),
                                              reshape_size=reshape_size)

    else:
        raise NotImplementedError('make_training_data is not implemented for '
                                  'dimensionality {}'.format(dimensionality))

    return None

