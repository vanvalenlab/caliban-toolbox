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
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import numpy as np
import os
import stat
import sys

from deepcell_toolbox.utils.io_utils import list_npzs_folder
from tensorflow.python.keras import backend as K

def slice_npz_batches(full_npz_path, batch_size, save_dir):
    '''
    takes an npz and splits it into "batches" (usually, frames or
    zslices of 3D data) to make annotation of it manageable. Each
    split npz is named with the frames it contains so that pieces
    can be put back together to form a continuous dataset later.

    Should be used with 4D npzs in shape (frames, y, x, channels) with
    filenames 'X' (raw images) and 'y' (annotations)

    Inputs:
        full_npz_path: full path to npz, used to load npz as well as
            construct appropriate names for the sliced npzs
        batch_size: (int) number of frames to include in each sliced npz
        save_dir: path to folder where sliced npzs should be saved

    Output:
        None (npzs are created and saved while function runs)
    '''

    # split npz filename up to use later
    npz_name = os.path.split(full_npz_path)[1]
    npz_basename = os.path.splitext(npz_name)[0]

    # load raw and annotated arrays from npz
    full_npz = np.load(full_npz_path)
    full_X, full_y = full_npz['X'][()], full_npz['y'][()]

    # calculate how many files will be made with batch_size (round up)
    num_batches = math.ceil(full_X.shape[0]/batch_size)

    for batch in range(num_batches):

        # starting and ending frames
        batch_start = batch*batch_size
        batch_end = (batch+1)*batch_size

        # if last set of frames, don't overshoot size of array
        if batch_end > full_X.shape[0]:
            batch_end = full_X.shape[0]

        # make new arrays with selected range of raw and annotated images
        partial_X = full_X[batch_start:batch_end,:,:,:]
        partial_y = full_y[batch_start:batch_end,:,:,:]

        # create appropriate new npz filenames
        if batch_size > 1:
            piece_name = "{0}_frames_{1}-{2}.npz".format(npz_basename, batch_start, batch_end-1)
        elif batch_size == 1:
            piece_name = "{0}_frames_{1}.npz".format(npz_basename, batch_start)

        # save sliced npz into save_dir

        # TODO: save empty pieces into subfolder in save_dir as in reshape_npz
        piece_path = os.path.join(save_dir, piece_name)
        np.savez(piece_path, X = partial_X, y = partial_y)

    return None


def reshape_npz(full_npz_path, x_size, y_size, save_dir, save_individual = True):
    '''
    Reshape a 4D npz (or 5D, if first dimension is 1) to have
    smaller x and y dimensions. There is no overlap between each
    reshaped piece.

    Inputs:
        full_npz: the full-sized npz (saved as X and y)
        x_size: new x size
        y_size: new y size
        save_dir: where individual reshaped npz files should be saved
        save_individual: bool, decides whether to split out each batch as individual
            new npz file (recommended for fig8), or to save as one 5D file containing
            batch info (could be repurposed for making training data)

    Output:
        Returns None.
        Saves either: reshaped (5D) npz in shape (batches, frames, y, x, channels)
            or n individual reshaped 4D npzs in shape (frames, y, x, channels)

    '''

    #create save_dir if it doesn't already exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(save_dir, mode)

    #get info from filename
    npz_name = os.path.split(full_npz_path)[1]
    npz_basename = os.path.splitext(npz_name)[0]

    #load npz
    full_npz = np.load(full_npz_path)

    # use variable names to avoid confusion with y and x dimensions
    full_raw, full_ann = full_npz['X'][()], full_npz['y'][()]

    #squeeze out batch dimension if needed
    if len(full_raw.shape) == 5:
        full_raw = np.squeeze(full_raw, axis = 0)
        full_ann = np.squeeze(full_ann, axis = 0)

    #figure out how many pieces are needed
    y_reps = math.ceil(full_raw.shape[1]/y_size)
    x_reps = math.ceil(full_raw.shape[2]/x_size)

    if not save_individual:
        # create arrays to hold all batches if not saving individually
        all_reshaped_raw_shape = (y_reps * x_reps, full_raw.shape[0], y_size, x_size, full_raw.shape[3])
        all_reshaped_ann_shape = (y_reps * x_reps, full_ann.shape[0], y_size, x_size, full_ann.shape[3])

        all_reshaped_raw = np.zeros(reshaped_raw_shape, dtype=K.floatx())
        all_reshaped_ann = np.zeros(reshaped_ann_shape, dtype = full_ann.dtype)

        # counter tracks which index of batch the reshaped array is put into
        counter = 0

    else:
        # set shape for 4D array that individual reshaped pieces will be saved as
        single_reshaped_raw_shape = (full_raw.shape[0], y_size, x_size, full_raw.shape[3])
        single_reshaped_ann_shape = (full_ann.shape[0], y_size, x_size, full_ann.shape[3])

    # loop over both x and y pieces
    for i in range(x_reps):
        for j in range(y_reps):

            # increment start and end indices for piece
            if i != x_reps - 1:
                x_start, x_end = i * x_size, (i + 1) * x_size
            # if it is the last piece in the row, don't overshoot array size
            # (even at the expense of overlapping with the previous piece)
            else:
                x_start, x_end = -x_size, full_raw.shape[2]

            # increment start and end indices for piece
            if j != y_reps - 1:
                y_start, y_end = j * y_size, (j + 1) * y_size
            # if it is the last piece in the row, don't overshoot array size
            # (even at the expense of overlapping with the previous piece)
            else:
                y_start, y_end = -y_size, full_raw.shape[1]

            # if you just want to reshape the npz without splitting it up,
            # each piece gets stored as a different batch
            if not save_individual:
                all_reshaped_raw[counter] = full_raw[:, y_start:y_end, x_start:x_end,:]
                all_reshaped_ann[counter] = full_ann[:, y_start:y_end, x_start:x_end,:]
                counter += 1

            # if splitting into separate files, do so as you go
            else:
                reshaped_raw = np.zeros(single_reshaped_raw_shape, dtype=K.floatx())
                reshaped_ann = np.zeros(single_reshaped_ann_shape, dtype = full_ann.dtype)

                reshaped_raw = full_raw[:, y_start:y_end, x_start:x_end,:]
                reshaped_ann = full_ann[:, y_start:y_end, x_start:x_end,:]

                #filename manipulations
                piece_name = "{0}_x_{1:02d}_y_{2:02d}.npz".format(npz_basename, i, j)
                piece_path = os.path.join(save_dir, piece_name)

                #don't save anything if npz piece is empty of annotations
                cells = np.unique(reshaped_ann)

                # background will be counted as one "cell" by np.unique
                if len(cells) > 1:
                    np.savez(piece_path, X = reshaped_raw, y = reshaped_ann)

                # don't save if the annotations are empty: Caliban can't open it

                # TODO: save empty npzs into subfolder in save_dir; there may be parts
                # of these files that should get annotated even if they are blank now
                else:
                    print('Piece of {0} from y: {1} to {2} and x: {3} to {4} is empty of annotations'.format(
                        full_npz_path, y_start, y_end, x_start, x_end))

    # save the 5D reshaped array once all the pieces are in it
    if not save_individual:
        reshaped_name = "{0}_reshaped_{1}_{2}.npz".format(npz_basename, y_size, x_size)
        reshaped_path = os.path.join(save_dir, reshaped_name)
        np.savez(reshaped_path, X = all_reshaped_raw, y = all_reshaped_ann)

    return None


def slice_npz_folder(src_folder, batch_size, save_dir):
    '''
    Use with slice_npz_batches to process a folder of npzs.
    Intended to be used after reshaping an npz (and saving as individual
    files). If only slicing a single npz, slice_npz_batches should be used.

    Inputs:
        src_folder: full path to folder that contains npzs to be sliced.
            Npzs should be 4D so that slice_npz_batches can run correctly.
        batch_size: (int) number of frames to include in each sliced npz
        save_dir: path to folder where sliced npzs should be saved

    Outputs:
        None
    '''
    # create the save dir if necessary
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        mode = stat.S_IRWXO | stat.S_IRWXU | stat.S_IRWXG
        os.chmod(save_dir, mode)

    # get list of npz files to slice
    npz_list = list_npzs_folder(src_folder)

    # call slice_npz_batches on each npz
    for npz in npz_list:
        full_npz_path = os.path.join(src_folder, npz)
        print('slicing', full_npz_path)
        slice_npz_batches(full_npz_path, batch_size, save_dir)

    return None


def relabel_npz_unique(full_npz_path, start_val = 1):
    '''
    Load an npz and relabel each feature so that every cell in every
    frame has a different label. Does NOT preserve relationships between
    frames! Intended for case when only segmentations need to be corrected;
    allows for easier use of caliban shortcuts (eg, replacing labels in
    Caliban does not accidentally lead to duplicate labels elsewhere in npz).
    Starts labeling at 1 by default and does not skip labels.

    Npz should be 4D in shape (frames, y, x, channels).

    Inputs:
        full_npz_path: full path to npz to relabel. Used to load and then
            save npz.
        start_val: What label to begin relabeling with. 1 by default but
            could be set to another value if needed for other purposes

    Returns:
        None (npz is overwritten with uniquely relabeled npz)

    '''
    # load npz
    npz = np.load(full_npz_path)

    # load raw and annotations, but we only need to modify annotations
    raw = npz['X'][()]
    annotations = npz['y'][()]

    # TODO: make sure that npz is 4D

    # assumes channels_last
    features = annotations.shape[-1]

    # create new array to store the relabeled annotations
    relabeled_annotations = np.zeros(annotations.shape, dtype = annotations.dtype)

    # features should be relabeled independently of each other
    for f in range(features):

        # how many unique cells in feature so far
        counter = 0

        # relabel each frame
        for frame in range(annotations.shape[0]):
            img = annotations[frame,:,:,f]

            # get all unique values in this frame, excluding 0 (background)
            unique_cells = np.unique(img)
            unique_cells = unique_cells[np.nonzero(unique_cells)]

            # create array from starting value to starting value + number of cells to relabel
            # ensures no labels are skipped
            relabel_ids = np.arange(start_val + counter, len(unique_cells) + start_val + counter)

            # populate relabeled_annotations array with relabeled annotations for that frame
            for cell_id, relabel_id in zip(unique_cells, relabel_ids):
                relabeled_annotations[frame,:,:,f] = np.where(img == cell_id, relabel_id,
                    relabeled_annotations[frame,:,:,f])

            # keep track of what label the next frame should start on
            counter += len(unique_cells)

    # overwrite original npz with relabeled npz (only the annotations have changed)
    np.savez(full_npz_path, X = raw, y = relabeled_annotations)

    return None


def relabel_npz_preserve_relationships():
    '''
    Placeholder for a function that relabels each feature in a 4D npz
    while preserving relationships. Eg, if cell 5 gets relabeled to cell 4,
    every instance of cell 5 in the movie will get relabeled to cell 4 as well.

    '''
    return None


def relabel_npzs_folder(npz_dir, relabel_type = 'unique'):
    '''
    relabel each npz in a folder with a given relabling strategy
    '''

    npz_list = list_npzs_folder(npz_dir)

    for npz in npz_list:
        full_npz_path = os.path.join(npz_dir, npz)

        if relabel_type == 'preserve':
            pass

        elif relabel_type == 'unique':
            relabel_npz_unique(full_npz_path)
