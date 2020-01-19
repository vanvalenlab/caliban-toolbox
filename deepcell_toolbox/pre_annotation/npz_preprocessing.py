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

    Returns:
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

        #don't save anything if npz piece is empty of annotations
        cells = np.unique(partial_y)

        # background will be counted as one "cell" by np.unique
        if len(cells) > 1:
            # save sliced npz into save_dir
            piece_path = os.path.join(save_dir, piece_name)
            np.savez(piece_path, X = partial_X, y = partial_y)

        # don't save if the annotations are empty: Caliban can't open it

        # TODO: save empty npzs into subfolder in save_dir; there may be parts
        # of these files that should get annotated even if they are blank now
        else:
            if batch_size > 1:
                print('Piece of {0} from frames {1} to {2} is empty of annotations'.format(
                full_npz_path, batch_start, batch_end))
            elif batch_size == 1:
                print('Piece of {0} at frame {1} is empty of annotations'.format(
                full_npz_path, batch_start))

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

    Returns:
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

    Returns:
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


def predict_zstack_cell_ids(img, next_img, threshold = 0.1):
    '''
    Predict labels for next_img based on intersection over union (iou)
    with img. If cells don't meet threshold for iou, they don't count as
    matching enough to share label with "matching" cell in img. Cells
    that don't have a match in img (new cells) get a new label so that
    output relabeled_next does not skip label values (unless label values
    present in prior image need to be skipped to avoid conflating labels).
    '''

    # relabel to remove skipped values, keeps subsequent predictions cleaner
    relabeled_img = np.zeros(next_img.shape, next_img.dtype)

    unique_cells = np.unique(next_img)
    unique_cells = unique_cells[np.nonzero(unique_cells)]

    # create array from starting value to starting value + number of cells to relabel
    # ensures no labels are skipped
    relabel_ids = np.arange(1, len(unique_cells) + 1)

    # populate relabeled_annotations array with relabeled annotations for that frame
    for cell_id, relabel_id in zip(unique_cells, relabel_ids):
        relabeled_img = np.where(next_img == cell_id, relabel_id,
                relabeled_img)

    next_img = relabeled_img

    #create np array that can hold all pairings between cells in one
    #image and cells in next image
    iou = np.zeros((np.max(img)+1, np.max(next_img)+1))

    vals = np.unique(img)
    cells = vals[np.nonzero(vals)]

    #nothing to predict off of
    if len(cells) == 0:
        print('skipped')
        return next_img

    next_vals = np.unique(next_img)
    next_cells = next_vals[np.nonzero(next_vals)]

    #no values to reassign
    if len(next_cells) == 0:
        return next_img

    #calculate IOUs
    for i in cells:
        for j in next_cells:
            intersection = np.logical_and(img==i,next_img==j)
            union = np.logical_or(img==i,next_img==j)
            iou[i,j] = intersection.sum(axis=(0,1)) / union.sum(axis=(0,1))

    #relabel cells appropriately

    #relabeled_next holds cells as they get relabeled appropriately
    relabeled_next = np.zeros(next_img.shape, dtype = np.uint16)

    #max_indices[cell_from_next_img] -> cell from first image that matches it best
    max_indices = np.argmax(iou, axis = 0)

    #put cells that into new image if they've been matched with another cell

    #keep track of which (next_img)cells don't have matches
    #this can be if (next_img)cell matched background, or if (next_img)cell matched
    #a cell already used
    unmatched_cells = []
    #don't reuse cells (if multiple cells in next_img match one particular cell)
    used_cells_src = []

    #next_cell ranges between 0 and max(next_img)
    #matched_cell is which cell in img matched next_cell the best

    # this for loop does the matching between cells
    for next_cell, matched_cell in enumerate(max_indices):
        #if more than one match, look for best match
        #otherwise the first match gets linked together, not necessarily reproducible

        # matched_cell != 0 prevents adding the background to used_cells_src
        if matched_cell != 0 and matched_cell not in used_cells_src:
            bool_matches = np.where(max_indices == matched_cell)
            count_matches = np.count_nonzero(bool_matches)
            if count_matches > 1:
                #for a given cell in img, which next_cell has highest iou
                matching_next_options = np.argmax(iou, axis =1)
                best_matched_next = matching_next_options[matched_cell]

                #ignore if best_matched_next is the background
                if best_matched_next != 0:
                    if next_cell != best_matched_next:
                        unmatched_cells = np.append(unmatched_cells, next_cell)
                        continue
                    else:
                        # don't add if bad match
                        if iou[matched_cell][best_matched_next] > threshold:
                            relabeled_next = np.where(next_img == best_matched_next, matched_cell, relabeled_next)

                        # if it's a bad match, we still need to add next_cell back into relabeled next later
                        elif iou[matched_cell][best_matched_next] <= threshold:
                            unmatched_cells = np.append(unmatched_cells, best_matched_next)

                        # in either case, we want to be done with the "matched_cell" from img
                        used_cells_src = np.append(used_cells_src, matched_cell)

            # matched_cell != 0 is still true
            elif count_matches == 1:
                #add the matched cell to the relabeled image
                if iou[matched_cell][next_cell] > threshold:
                    relabeled_next = np.where(next_img == next_cell, matched_cell, relabeled_next)
                else:
                    unmatched_cells = np.append(unmatched_cells, next_cell)

                used_cells_src = np.append(used_cells_src, matched_cell)

        elif matched_cell in used_cells_src and next_cell != 0:
            #skip that pairing, add next_cell to unmatched_cells
            unmatched_cells = np.append(unmatched_cells, next_cell)

        #if the cell in next_img didn't match anything (and is not the background):
        if matched_cell == 0 and next_cell !=0:
            unmatched_cells = np.append(unmatched_cells, next_cell)
            #note: this also puts skipped (nonexistent) labels into unmatched cells, main reason to relabel first

    #figure out which labels we should use to label remaining, unmatched cells

    #these are the values that have already been used in relabeled_next
    relabeled_values = np.unique(relabeled_next)[np.nonzero(np.unique(relabeled_next))]

    #to account for any new cells that appear, create labels by adding to the max number of cells
    #assumes that these are new cells and that all prev labels have been assigned
    #only make as many new labels as needed

    if len(relabeled_values) > 0:
        current_max = max(np.max(cells), np.max(relabeled_values)) + 1
    else:
        current_max = np.max(cells) + 1

    stringent_allowed = []
    for additional_needed in range(len(unmatched_cells)):
        stringent_allowed.append(current_max)
        current_max += 1

    #replace each unmatched cell with a value from the stringent_allowed list,
    #add that relabeled cell to relabeled_next
    if len(unmatched_cells) > 0:
        for reassigned_cell in range(len(unmatched_cells)):
            relabeled_next = np.where(next_img == unmatched_cells[reassigned_cell],
                                 stringent_allowed[reassigned_cell], relabeled_next)

    return relabeled_next


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


def relabel_npz_preserve_relationships(full_npz_path, start_val = 1):
    '''
    Relabels each feature in a 4D npz while preserving relationships.
    Eg, if cell 5 gets relabeled to cell 4, every instance of cell 5 in
    the movie will get relabeled to cell 4 as well.

    Npz should be 4D in shape (frames, y, x, channels).

    Inputs:
        full_npz_path: full path to npz to relabel. Used to load and then
            save npz.
        start_val: What label to begin relabeling with. 1 by default but
            could be set to another value if needed for other purposes

    Returns:
        None (relabeled npzs are saved in place)

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

        img_stack = annotations[:,:,:,f]

        # get all unique values in this stack, excluding 0 (background)
        unique_cells = np.unique(img_stack)
        unique_cells = unique_cells[np.nonzero(unique_cells)]

        # create array from starting value to starting value + number of cells to relabel
        # ensures no labels are skipped
        relabel_ids = np.arange(start_val, len(unique_cells) + start_val)

        # populate relabeled_annotations array with relabeled annotations for that frame
        for cell_id, relabel_id in zip(unique_cells, relabel_ids):
            relabeled_annotations[:,:,:,f] = np.where(img_stack == cell_id, relabel_id,
                relabeled_annotations[:,:,:,f])

    # overwrite original npz with relabeled npz (only the annotations have changed)
    np.savez(full_npz_path, X = raw, y = relabeled_annotations)

    return None


def relabel_npz_all_frames(full_npz_path, start_val = 1):
    '''
    Relabels each frame in an npz starting from start_val. Each frame gets relabeled
    independently of the other frames, so this can scramble relationship data if it
    exists. Useful for cases where relationships do not need to be preserved (eg,
    annotating, correcting, possibly reshaping npzs full of 2D annotations).

    Npz should be 4D in shape (frames, y, x, channels).

    Inputs:
        full_npz_path: full path to npz to relabel. Used to load and then
            save npz.
        start_val: What label to begin relabeling with. 1 by default but
            could be set to another value if needed for other purposes

    Returns:
        None (relabeled npzs are saved in place)
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

        # relabel each frame
        for frame in range(annotations.shape[0]):
            img = annotations[frame,:,:,f]

            # get all unique values in this frame, excluding 0 (background)
            unique_cells = np.unique(img)
            unique_cells = unique_cells[np.nonzero(unique_cells)]

            # create array from starting value to starting value + number of cells to relabel
            # ensures no labels are skipped
            relabel_ids = np.arange(start_val, len(unique_cells) + start_val)

            # populate relabeled_annotations array with relabeled annotations for that frame
            for cell_id, relabel_id in zip(unique_cells, relabel_ids):
                relabeled_annotations[frame,:,:,f] = np.where(img == cell_id, relabel_id,
                    relabeled_annotations[frame,:,:,f])

    # overwrite original npz with relabeled npz (only the annotations have changed)
    np.savez(full_npz_path, X = raw, y = relabeled_annotations)

    return None


def relabel_npz_zstack_prediction(full_npz_path, start_val = 1, threshold = 0.1):
    '''
    Relabels the first frame of an npz starting from start_val. Each subsequent frame
    gets relabeled based on iou (intersection over union) with the previous frame.
    This can scramble human-annotated 3D label assignments if they exist--
    use relabel_npz_preserve_relationships for relabeling human-corrected 3D labels.
    Useful as a pre-processing step to reduce the human labor that goes into assigning labels.
    This step should be applied when:
        - launching a 3D npz that was created from unrelated 2D predictions
        - the npz has not already been relabeled
        - deep learning predictions are not robust for the data type yet

    Note: the iou-based prediction was implemented for zstack prediction, but should
    work okay for timelapse movies as well (although it is not expected to get divisions).
    Deep learning tracking should be used to predict timelapse movies if the segmentations
    are sufficiently good, otherwise this is an acceptable first pass.

    Npz should be 4D in shape (frames, y, x, channels).

    Inputs:
        full_npz_path: full path to npz to relabel. Used to load and then
            save npz.
        start_val: What label to begin relabeling with. 1 by default but
            could be set to another value if needed for other purposes
        threshold: iou between cells that must be reached for zstack prediction to
            consider them matched. 0.1 by default

    Returns:
        None (relabeled npzs are saved in place)
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

        # relabel the first frame
        # (even if this is empty, each next_frame gets relabeled as first step in
        # predict_zstack_cell_ids, so whichever frame is the first to have labels
        # will be relabeled from 1)

        first_img = annotations[0,:,:,f]
        unique_cells = np.unique(first_img)
        unique_cells = unique_cells[np.nonzero(unique_cells)]

        relabel_ids = np.arange(start_val, len(unique_cells) + start_val)
        for cell_id, relabel_id in zip(unique_cells, relabel_ids):
                        relabeled_annotations[0,:,:,f] = np.where(first_img == cell_id, relabel_id,
                            relabeled_annotations[0,:,:,f])

        # from that frame onwards, predict the rest
        for frame in range(annotations.shape[0] -1):

            # base labels off of this frame
            img = relabeled_annotations[frame,:,:,f]

            # this is frame that gets relabeled with predictions
            next_img = annotations[frame + 1, :,:,f]

            # put predictions into relabeled_annotations
            predicted_next = predict_zstack_cell_ids(img, next_img, threshold)
            relabeled_annotations[frame + 1, :,:,f] = predicted_next

    # overwrite original npz with relabeled npz (only the annotations have changed)
    np.savez(full_npz_path, X = raw, y = relabeled_annotations)

    return None


def relabel_npzs_folder(npz_dir, relabel_type = 'preserve', start_val = 1, threshold = 0.1):
    '''
    Relabel each npz in a folder with a given relabling strategy.
    Relableling strategy is 'preserve' by default so that relationships
    are not lost/scrambled unless user intentionally uses different strategy.
    Note: "predict" relabeling scheme does not use deep learning predictions and
    will not produce lineage predictions. This option is a computer vision approach
    to help annotators when a deep learning prediction is not available.

    Inputs:
        npz_dir: full path to directory holding npzs to be relabeled. All npzs
            in dir will be relabeled in place.
        relabel_type: string to choose which relabeling function to call. Options
            are:
                preserve - preserve 3D relationships of labels
                unique - each cell in each frame gets a unique label
                all_frames - each frame is relabeled from start_val, independent of
                    other frame labeling
                predict - relabel first frame to start from 1, then predict labels in
                    subsequent frames by iou comparisons between each frame and the next frame
        start_val: which starting value to pass to relabeling function. 1 by default
        threshold: iou between cells that must be reached for zstack prediction to
            consider them matched. Default is 0.1, only used in relabel_npz_zstack_prediction

    Returns:
        None. Npzs are relabeled in place as function runs
    '''

    npz_list = list_npzs_folder(npz_dir)

    for npz in npz_list:
        full_npz_path = os.path.join(npz_dir, npz)

        if relabel_type == 'preserve':
            relabel_npz_preserve_relationships(full_npz_path, start_val)

        elif relabel_type == 'unique':
            relabel_npz_unique(full_npz_path, start_val)

        elif relabel_type == 'all_frames':
            relabel_npz_all_frames(full_npz_path, start_val)

        elif relabel_type == 'predict':
            relabel_npz_zstack_prediction(full_npz_path, start_val, threshold)


def compute_crop_indices(img_len, crop_size, overlap_frac):
    """ Determine how to crop the image.

    Inputs
        img_len: length of the image to be cropped
        crop_size: size in pixels of the crop
        overlap_frac: fraction that adjacent crops will overlap each other

    Outputs:
        start_indices: array of row coordinates for crop starts
        end_indices: array of coordinates for crop ends
        padding: number of pixels of padding at start and end of image
    """

    # compute overlap fraction in pixels
    overlap_pix = math.floor(crop_size * overlap_frac)

    # crops start at pixel 0 (from padded image)
    start_indices = np.arange(0, img_len, crop_size)

    # crops end at crop_size + 2 * overlap pixels away from the start
    end_indices = start_indices + (crop_size + 2 * overlap_pix)

    padding = end_indices[-1] - (img_len + overlap_pix)

    return start_indices, end_indices, (overlap_pix, padding)


def crop_images(input_data, row_start, row_end, col_start, col_end, padding):
    crop_num = len(row_start) * len(col_start)
    crop_size_row = row_end[0] - row_start[0]
    crop_size_col = col_end[0] - col_start[0]

    cropped_stack = np.zeros((crop_num, crop_size_row, crop_size_col, input_data.shape[2]))

    padded_input = np.pad(input_data, padding, mode="constant", constant_values=0)
    crop_counter = 0
    for i in range(len(row_start)):
        for j in range(len(col_start)):
            cropped_stack[crop_counter, ...] = padded_input[row_start[i]:row_end[i], col_start[i]:col_end[i], :]

    return cropped_stack, padded_input.shape


def save_npz(cropped_x_data, cropped_y_data, file_base, save_dir):

    for i in range(cropped_x_data.shape[0]):
        np.savez(os.path.join(save_dir, "{}_crop_{}.npz".format(file_base, i)),
                              X=cropped_x_data[i, ...], y=cropped_y_data[i, ...])


def crop_npz(npz_name, base_dir, crop_size, overlap_frac):
    npz = np.load(os.path.join(base_dir, npz_name))
    X = npz["X"]
    y = npz["y"]

    row_start, row_end, row_padding = compute_crop_indices(img_len=X.shape[0], crop_size=crop_size[0],
                                                                           overlap_frac=overlap_frac)

    col_start, col_end, col_padding = compute_crop_indices(img_len=X.shape[1], crop_size=crop_size[1],
                                                           overlap_frac=overlap_frac)

    X_cropped, padded_shape = crop_images(X, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end,
                            padding=(row_padding, col_padding, (0, 0)))

    y_cropped, padded_shape = crop_images(y, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end,
                            padding=(row_padding, col_padding, (0, 0)))

    save_npz(cropped_x_data=X_cropped, cropped_y_data=y_cropped, file_base=npz_name, save_dir=base_dir)