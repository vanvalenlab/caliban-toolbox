import copy
import importlib

from caliban_toolbox.pre_annotation import relabel
from skimage.segmentation import relabel_sequential

import numpy as np

importlib.reload(relabel)


def test_relabel_preserve_relationships():
    stack = np.zeros((1, 5, 1, 1, 100, 100, 1))
    base_frame = np.zeros((100, 100))

    base_frame[2:10, 1:9] = 10
    base_frame[80:85, 10:30] = 20
    base_frame[50:60, 20:30] = 30
    base_frame[2:10, 20:30] = 40
    base_frame[70:80, 10:30] = 5
    base_frame[20:30, 90:94] = 6
    base_frame[40:50, 60:80] = 90

    # selectively remove one random cell from each frame
    for i in range(stack.shape[1]):
        dropout_val = np.random.choice(np.unique(base_frame))
        temp_frame = copy.copy(base_frame)
        temp_frame[temp_frame == dropout_val] = 0
        stack[0, i, 0, 0, :, :, 0] = temp_frame

    relabeled_stack = relabel.relabel_preserve_relationships(stack)

    # select one of the frames to use a starting point to identify mapping
    stack_idx = 2

    for cell in np.unique(relabeled_stack[0, stack_idx]):
        # figure out index of original cell that overlaps with this cell
        cell_mask = relabeled_stack[0, stack_idx, 0, 0, :, :, 0] == cell
        original_idx = stack[0, stack_idx, 0, 0, cell_mask, 0][0]

        # make sure all instances of this cell are same as all instances of original cell
        assert np.all(np.equal(relabeled_stack == cell, stack == original_idx))

    # number of unique IDs in original stack is equal to max of relabeled stack + 1 (for 0 label)
    assert int(np.max(relabeled_stack) + 1) == len(np.unique(stack))


def test_relabel_all_frames():
    fov_len, stack_len, num_crops, num_slices, rows, cols, channels = 2, 5, 3, 11, 100, 100, 1

    stack = np.zeros((fov_len, stack_len, num_crops, num_slices, rows, cols, channels))
    stack[:, :, :, :, 2:10, 1:9, 0] = 10
    stack[:, :, :, :, 80:85, 10:30, 0] = 20
    stack[:, :, :, :, 50:60, 20:30, 0] = 30
    stack[:, :, :, :, 2:10, 20:30, 0] = 40
    stack[:, :, :, :, 70:80, 10:30, 0] = 5
    stack[:, :, :, :, 20:30, 90:94, 0] = 6
    stack[:, :, :, :, 40:50, 60:80, 0] = 90

    relabled_stack = relabel.relabel_all_frames(stack)

    for fov in range(fov_len):
        for stack in range(stack_len):
            for crop in range(num_crops):
                for slice in range(num_slices):
                    current_crop = relabled_stack[fov, stack, crop, slice, :, :, 0]
                    assert int(np.max(current_crop) + 1) == len(np.unique(current_crop))



