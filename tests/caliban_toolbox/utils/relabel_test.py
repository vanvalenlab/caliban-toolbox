import copy
import importlib

from caliban_toolbox.pre_annotation import relabel
from skimage.segmentation import relabel_sequential

import numpy as np

importlib.reload(relabel)


def test_relabel_preserve_relationships():
    stack = np.zeros((5, 100, 100, 1))
    base_frame = np.zeros((100, 100))

    base_frame[2:10, 1:9] = 10
    base_frame[80:85, 10:30] = 20
    base_frame[50:60, 20:30] = 30
    base_frame[2:10, 20:30] = 40
    base_frame[70:80, 10:30] = 5
    base_frame[20:30, 90:94] = 6
    base_frame[40:50, 60:80] = 90

    # selectively remove one random cell from each frame
    for i in range(stack.shape[0]):
        dropout_val = np.random.choice(np.unique(base_frame))
        temp_frame = copy.copy(base_frame)
        temp_frame[temp_frame == dropout_val] = 0
        stack[i, ..., 0] = temp_frame

    relabeled_stack = relabel.relabel_npz_preserve_relationships(stack)

    for cell in np.unique(relabeled_stack[2]):
        # figure out corresponding value in original stack
        cell_mask = relabeled_stack[2, :, :, 0] == cell
        original_idx = stack[2, cell_mask, 0][0]

        # make sure all cells have same values
        assert np.all(np.equal(relabeled_stack == cell, stack == original_idx))


