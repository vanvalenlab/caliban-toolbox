from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import random

from imageio import imread
import numpy as np
from tensorflow.python.keras import backend as K
from deepcell_toolbox.utils.io_utils import get_img_names, get_image


CHANNELS_FIRST = K.image_data_format() == 'channels_first'


def post_annotation_load_training_images_3d(training_dir,
                            training_folders,
                            channel_names,
                            image_size,
                            num_frames):
    """Load each image in the training_direcs into a numpy array.

    Args:
        direc_name: directory containing folders of training data
        training_direcs: list of directories of images inside direc_name.
        raw_image_direc: directory name inside each training dir with raw images
        channel_names: Loads all raw images with a channel_name in the filename
        image_size: size of each image as tuple (x, y)

    Returns:
        4D tensor of image data
    """
    is_channels_first = K.image_data_format() == 'channels_first'
    # Unpack size tuples
    image_size_x, image_size_y = image_size

    num_batches = len(training_folders)

    if num_frames == None:
        num_frames = len(get_img_names(os.path.join(training_dir, tranining_folders[0], channel_names[0])))

    # Initialize training data array
    if K.image_data_format() == 'channels_first':
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
                             num_frames):
    """Load each annotated image in the training_direcs into a numpy array.

    Args:
        direc_name: directory containing folders of training data
        training_direcs: list of directories of images inside direc_name.
        annotation_direc: directory name inside each training dir with masks
        annotation_name: Loads all masks with annotation_name in the filename
        image_size: size of each image as tuple (x, y)

    Returns:
        4D tensor of label masks
    """
    is_channels_first = K.image_data_format() == 'channels_first'
    # Unpack size tuple
    image_size_x, image_size_y = image_size

    # wrapping single annotation name in list for consistency
    if not isinstance(annotation_folders, list):
        annotation_folders = [annotation_folders]
    
    num_batches = len(training_folders)
    
    if num_frames == None:
        num_frames = len(get_img_names(os.path.join(training_dir, annotation_folders[0], channel_names[0])))        

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
    """Read all images in training directories and save as npz file.

    Args:
        direc_name: directory containing folders of training data
        file_name_save: full filepath for npz file where the data will be saved
        training_direcs: directories of images located inside direc_name.
                         If None, all directories in direc_name are used.
        raw_image_direc: directory name inside each training dir with raw images
        annotation_direc: directory name inside each training dir with masks
        channel_names: Loads all raw images with a channel_name in the filename
        annotation_name: Loads all masks with annotation_name in the filename
        reshape_size: If provided, will reshape the images to the given size
    """
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
                            image_size):
    """Load each image in the training_direcs into a numpy array.

    Args:
        direc_name: directory containing folders of training data
        training_direcs: list of directories of images inside direc_name.
        raw_image_direc: directory name inside each training dir with raw images
        channel_names: Loads all raw images with a channel_name in the filename
        image_size: size of each image as tuple (x, y)

    Returns:
        4D tensor of image data
    """
    is_channels_first = K.image_data_format() == 'channels_first'
    # Unpack size tuples
    image_size_x, image_size_y = image_size

    num_batches = len(get_img_names(os.path.join(training_dir, channel_names[0])))

    # Initialize training data array
    if K.image_data_format() == 'channels_first':
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
                             image_size):
    """Load each annotated image in the training_direcs into a numpy array.

    Args:
        direc_name: directory containing folders of training data
        training_direcs: list of directories of images inside direc_name.
        annotation_direc: directory name inside each training dir with masks
        annotation_name: Loads all masks with annotation_name in the filename
        image_size: size of each image as tuple (x, y)

    Returns:
        4D tensor of label masks
    """
    is_channels_first = K.image_data_format() == 'channels_first'
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
    """Read all images in training directories and save as npz file.

    Args:
        direc_name: directory containing folders of training data
        file_name_save: full filepath for npz file where the data will be saved
        training_direcs: directories of images located inside direc_name.
                         If None, all directories in direc_name are used.
        raw_image_direc: directory name inside each training dir with raw images
        annotation_direc: directory name inside each training dir with masks
        channel_names: Loads all raw images with a channel_name in the filename
        annotation_name: Loads all masks with annotation_name in the filename
        reshape_size: If provided, will reshape the images to the given size
    """
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

