from filters import label
import numpy as np


def labels_map(lb0, lb1):
    """lb0 and lb1 should have objects in the same locations but different labels.
    """
    lbnums = np.unique(lb0).tolist()
    lbnums.remove(0)
    st = []
    for n0 in lbnums:
        n_sets = set(lb1[lb0 == n0])
        assert not len(n_sets) == 0
        n1 = list(n_sets)[0]
        st.append((n0, n1))
    return st


def convert_labels(lb_ref_to, lb_ref_from, lb_convert):
    """
    lb_ref_to: 
    lb_ref_from: 
    lb_convert: a labeled image to be converted.
    """
    lbmap_to, lbmap_from = zip(*labels_map(lb_ref_to, lb_ref_from))
    arr = np.zeros(lb_convert.shape, dtype=np.uint16)
    lb = lb_convert.copy()

    for n0, n1 in zip(lbmap_to, lbmap_from):
        arr[lb == n1] = n0
        lb[lb == n1] = 0

    remained = label(lb)
    remained = remained + arr.max()
    remained[remained == arr.max()] = 0
    arr = arr + remained
    return arr