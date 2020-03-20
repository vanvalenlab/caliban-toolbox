from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import numpy as np
import os
import stat

from caliban_toolbox.utils.io_utils import list_npzs_folder
from skimage.segmentation import relabel_sequential


def relabel_preserve_relationships(annotations, start_val=1):
    """Relabels annotations while preserving relationships within in each stack. Eg, if cell 5 gets relabeled to cell 4,
    every instance of cell 5 in the stack will get relabeled to cell 4 as well.

    Inputs
        annotations: xarray of data to relabel [fovs, stacks, crops, slices, rows, cols, channels]
        start_val: Value where relabeling will begin

    Returns:
        relabeled_annotations: xarray containing annotations that have been relabeled """

    fov_len, stack_len, num_crops, num_slices, rows, cols, channels = annotations.shape

    if num_crops > 1:
        raise ValueError("Relabeling should occur before cropping or after reconstruction")

    if num_slices > 1:
        raise ValueError("Relabeling occur before slicing or after reconstruction")

    # create new array to store the relabeled annotations
    relabeled_annotations = np.zeros(annotations.shape, dtype = annotations.dtype)

    # get all unique values in this stack, excluding 0 (background)
    unique_cells = np.unique(annotations)
    unique_cells = unique_cells[np.nonzero(unique_cells)]

    # create array to hold sequential relabeled ids
    relabel_ids = np.arange(start_val, len(unique_cells) + start_val)

    # populate relabeled_annotations array corresponding relabeled ids
    for cell_id, relabel_id in zip(unique_cells, relabel_ids):
        relabeled_annotations = np.where(annotations == cell_id, relabel_id, relabeled_annotations)

    # overwrite original npz with relabeled npz (only the annotations have changed)
    return relabeled_annotations


def relabel_all_frames(input_data, start_val=1):
    """Relabels all frames in all montages in all fovs independently from start_val.

    Inputs
        input_data: array of [fovs, stacks, crops, slices, rows, cols, channels] to be relabeled
        start_val: Value of first label in each frame

    Returns:
        relabeled_data: array of relabeled data"""

    relabeled_annotations = np.zeros(input_data.shape)
    fov_len, stack_len, num_crops, num_slices, _, _, _ = input_data.shape

    # relabel each frame
    for fov in range(fov_len):
        for stack in range(stack_len):
            for crop in range(num_crops):
                for slice in range(num_slices):
                    img = input_data[fov, stack, crop, slice, :, :, 0]
                    img_relabeled, _, _ = relabel_sequential(img, start_val)
                    relabeled_annotations[fov, stack, crop, slice, :, :, 0] = img_relabeled

    return relabeled_annotations


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
    relabeled_next = np.zeros(next_img.shape, dtype=np.uint16)

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
