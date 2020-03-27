import numpy as np

from skimage.segmentation import relabel_sequential


def relabel_preserve_relationships(annotations, start_val=1):
    """Relabels annotations while preserving relationships within in each stack.

    Eg, if cell 5 gets relabeled to cell 4, every instance of cell 5 in the stack will get relabeled to cell 4 as well.

    Args:
        annotations: xarray of data to relabel [fovs, stacks, crops, slices, rows, cols, channels]
        start_val: Value where relabeling will begin

    Returns:
        relabeled_annotations: xarray containing annotations that have been relabeled
    """

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

    Args:
        input_data: array of [fovs, stacks, crops, slices, rows, cols, channels] to be relabeled
        start_val: Value of first label in each frame

    Returns:
        relabeled_data: array of relabeled data
    """

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


def predict_relationships_helper(current_img, next_img, threshold=0.1):
    """Predict labels for next_img based on label values in img

    Cells that overlap at least "threshold" amount will be given the same label.
    Cells that don't meet this cutoff will given a unique label for that frame

    Args:
        current_img: current image that will be used to identify existing labels
        next_img: image to be relabeled based on overlap with img
        threshold: iou cutoff to determine if cells match

    Returns:
        next_relabeled: corrected version of next_img
    """

    # relabel to remove skipped values, keeps subsequent predictions cleaner
    next_img, _, _ = relabel_sequential(next_img)

    # get unique cells from each image
    current_cells = np.unique(current_img[current_img > 0])
    next_cells = np.unique(next_img[next_img > 0])

    # nothing to predict off of
    if len(current_cells) == 0:
        print('skipped')
        return next_img

    # no values to reassign
    if len(next_cells) == 0:
        return next_img

    # create np array that can hold all pairings between cells in both images
    iou = np.zeros((np.max(current_img) + 1, np.max(next_img) + 1))

    # calculate IOUs
    for i in current_cells:
        for j in next_cells:
            intersection = np.logical_and(current_img == i, next_img == j)
            union = np.logical_or(current_img == i, next_img == j)
            iou[i, j] = intersection.sum(axis=(0, 1)) / union.sum(axis=(0, 1))

    # relabeled_next holds cells as they get relabeled appropriately
    next_img_relabeled = np.zeros(next_img.shape, dtype=np.uint16)

    # max_indices[cell_from_next_img] -> cell from first image that matches it best
    max_indices = np.argmax(iou, axis=0)

    # next_cells without a good match, either due to overlap with background or already used cell
    next_cells_unmatched = []

    # current cells that been assigned a match with a next_cell
    current_cells_used = []

    # identify matching between each next_cell and current_cell
    for next_cell, current_cell_match in enumerate(max_indices):

        # next_cell matches a non-background current_cell that hasn't already been assigned elsewhere
        if current_cell_match != 0 and current_cell_match not in current_cells_used:

            num_matches = np.sum(max_indices == current_cell_match)

            # more than one next_cell has the same current_cell as best match
            if num_matches > 1:

                # for a given cell in current_img, which next_cell has highest iou
                max_indices_next = np.argmax(iou, axis=1)
                best_matched_next = max_indices_next[current_cell_match]

                # if the next_cell with the best match to current_cell_match is background, we skip this next_cell
                if best_matched_next == 0:
                    continue
                else:
                    # if this next_cell isn't the best match for current_cell_match, we add it to unmatched list
                    if next_cell != best_matched_next:
                        next_cells_unmatched = np.append(next_cells_unmatched, next_cell)
                        continue
                    else:
                        # if it's the best match and above the IOU threshold, we add it to the relabeled image
                        if iou[current_cell_match][next_cell] > threshold:
                            next_img_relabeled = np.where(next_img == next_cell, current_cell_match, next_img_relabeled)

                        # if it's a bad match, we add next_cell to unmatched list
                        elif iou[current_cell_match][next_cell] <= threshold:
                            next_cells_unmatched = np.append(next_cells_unmatched, next_cell)

                        # in either case, we want to be done with the "current_cell_match" from img
                        current_cells_used = np.append(current_cells_used, current_cell_match)

            # only a single match, check to see if IOU threshold is met
            elif num_matches == 1:
                if iou[current_cell_match][next_cell] > threshold:
                    next_img_relabeled = np.where(next_img == next_cell, current_cell_match, next_img_relabeled)
                else:
                    next_cells_unmatched = np.append(next_cells_unmatched, next_cell)

                current_cells_used = np.append(current_cells_used, current_cell_match)

        # the single current_cell_match for this next_cell has already been used
        elif current_cell_match in current_cells_used and next_cell != 0:
            # skip that pairing, add next_cell to next_cells_unmatched
            next_cells_unmatched = np.append(next_cells_unmatched, next_cell)

        # if the next_cell is not background, and did not match any nonzero current_cells
        if next_cell != 0 and current_cell_match == 0:
            next_cells_unmatched = np.append(next_cells_unmatched, next_cell)

    # Since both images were relabeled from 1, all values below the max in each are already used
    current_max = max(np.max(current_img), np.max(next_img_relabeled))

    # We increment from current_max to create new values for remaining cells that weren't matched
    vals_for_unmatched_cells = list(range(current_max + 1, current_max + 1 + len(next_cells_unmatched)))

    # relabel each unmatched cell with new value
    if len(next_cells_unmatched) > 0:
        for reassigned_cell in range(len(next_cells_unmatched)):
            next_img_relabeled = np.where(next_img == next_cells_unmatched[reassigned_cell],
                                          vals_for_unmatched_cells[reassigned_cell], next_img_relabeled)

    return next_img_relabeled


def predict_relationships(image_stack, start_val=1, threshold=0.1):
    """Predicts relationships between cells across different frames by using an IOU cutoff.

    Relabels the first frame starting from start_val. This will scramble human-annotated 3D label assignments. Use
    relabel_npz_preserve_relationships for relabeling human-corrected 3D labels.

    This step should be applied when:
        - launching a 3D npz that was created from unrelated 2D predictions
        - the npz has not already been relabeled
        - deep learning predictions are not robust for the data type yet

    Note: the iou-based prediction was implemented for zstack prediction, but should
    work okay for timelapse movies as well (although it is not expected to get divisions).

    Args:
        image_stack: xarray of [fovs, stacks, crops, slices, rows, cols, channels]
        start_val: Label value where labeling will begin.
        threshold: iou threshold for classifying cells as a match

    Returns:
        relabeled_stack: stack of relabeled images
    """

    # create new array to store the relabeled annotations
    relabeled_annotations = np.zeros(image_stack.shape, dtype=image_stack.dtype)

    # relabel the first frame
    first_img = image_stack[0, :, :, 0]
    unique_cells = np.unique(first_img)
    unique_cells = unique_cells[np.nonzero(unique_cells)]

    relabel_ids = np.arange(start_val, len(unique_cells) + start_val)
    for cell_id, relabel_id in zip(unique_cells, relabel_ids):
                    relabeled_annotations[0, :, :, 0] = np.where(first_img == cell_id, relabel_id,
                                                                 relabeled_annotations[0, :, :, 0])

    # from that frame onwards, predict the rest
    for frame in range(image_stack.shape[0] - 1):

        # base labels off of this frame
        img = relabeled_annotations[frame, :, :, 0]

        # this is frame that gets relabeled with predictions
        next_img = image_stack[frame + 1, :, :, 0]

        # put predictions into relabeled_annotations
        predicted_next = predict_relationships_helper(img, next_img, threshold)
        relabeled_annotations[frame + 1, :, :, 0] = predicted_next

    return relabeled_annotations


def relabel_data(input_data, relabel_type='preserve', start_val=1, threshold=0.1):
    """Relabel stacked labels for easier annotation in Caliban

    Args:
        input_data: xarray of data to be relabeled
        relabel_type: string to choose which relabeling function to call.
            preserve - preserve existing relationships of between labels in different frames
            all_frames - each frame is independently relabeled from start_val
            predict - attempt to link cells acrossframes via IOU for overlap

        start_val: lowest value for relabeling
        threshold: minimum iou threshold to count an overlap for predict relabeling

    Returns:
        array of relabel
    """

    allowed_relabels = ["preserve", "all_frames", "predict"]

    if relabel_type not in allowed_relabels:
        raise ValueError("relable_type must be one of [preserve, all_frames, predict]: got {}".format(relabel_type))

    if relabel_type == 'preserve':
        relabeled = relabel_preserve_relationships(input_data, start_val)

    elif relabel_type == 'all_frames':
        relabeled = relabel_all_frames(input_data, start_val)

    elif relabel_type == 'predict':
        relabeled = predict_relationships(input_data, start_val, threshold)

    return relabeled


