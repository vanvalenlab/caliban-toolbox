"""

Label labels1 to the same value from labels0. Not tracked is negative.
Turn labels0 into negative.

"""


from utils.filters import label
from utils.postprocess_utils import regionprops
from scipy.spatial.distance import cdist
from utils.track_utils import calc_massdiff, find_one_to_one_assign
from utils.global_holder import holder  # holder is used to store parameters that needs to be called many times
import numpy as np
from utils.track_utils import call_lap, pick_closer_cost
from utils.filters import labels2outlines
from utils.concave_seg import wshed_raw, CellCutter
from utils.track_utils import _find_best_neck_cut, _update_labels_neck_cut
from utils.global_holder import holder
from scipy.ndimage import gaussian_laplace, binary_dilation
from utils.binary_ops import grey_dilation
import logging

logger = logging.getLogger(__name__)

np.random.seed(0)


def nearest_neighbor(img0, img1, labels0, labels1, DISPLACEMENT=20, MASSTHRES=0.2):
    """
    labels0 and labels1: the positive values for non-tracked objects and the negative values for tracked objects.
    """
    labels = -labels1.copy()
    rps0 = regionprops(labels0, img0)
    rps1 = regionprops(labels1, img1)
    if not rps0 or not rps1:
        return labels0, labels
    dist = cdist([i.centroid for i in rps0], [i.centroid for i in rps1])
    massdiff = calc_massdiff(rps0, rps1)
    binary_cost = (dist < DISPLACEMENT) * (abs(massdiff) < MASSTHRES)
    idx1, idx0 = find_one_to_one_assign(binary_cost)
    for i0, i1 in zip(idx0, idx1):
        labels[labels1 == rps1[i1].label] = rps0[i0].label
        labels0[labels0 == rps0[i0].label] = -rps0[i0].label
    return labels0, labels


def nn_closer(img0, img1, labels0, labels1, DISPLACEMENT=30, MASSTHRES=0.25):
    labels = -labels1.copy()
    rps0 = regionprops(labels0, img0)
    rps1 = regionprops(labels1, img1)

    if not rps0 or not rps1:
        return labels0, labels

    dist = cdist([i.centroid for i in rps0], [i.centroid for i in rps1])
    massdiff = calc_massdiff(rps0, rps1)
    binary_cost = (dist < DISPLACEMENT) * (abs(massdiff) < MASSTHRES)
    binary_cost = pick_closer_cost(binary_cost, dist)
    binary_cost = pick_closer_cost(binary_cost.T, dist.T).T
    idx1, idx0 = find_one_to_one_assign(binary_cost)
    for i0, i1 in zip(idx0, idx1):
        labels[labels1 == rps1[i1].label] = rps0[i0].label
        labels0[labels0 == rps0[i0].label] = -rps0[i0].label
    return labels0, labels


def run_lap(img0, img1, labels0, labels1, DISPLACEMENT=30, MASSTHRES=0.2):
    '''Linear assignment problem for mammalian cells.
    Cost matrix is simply the distance.
    costDie and costBorn are variables changing over frame. Update it through holder.

    Args:
    DISPLACEMENT (int): The maximum distance (in pixel)
    MASSTHRES (float):  The maximum difference of total intensity changes.
                        0.2 means it allows for 20% total intensity changes.
    '''
    print('run_lap')
    labels = -labels1.copy()
    rps0 = regionprops(labels0, img0)
    rps1 = regionprops(labels1, img1)

    if not rps0 or not rps1:
        return labels0, labels

    dist = cdist([i.centroid for i in rps0], [i.centroid for i in rps1])
    massdiff = calc_massdiff(rps0, rps1)

    '''search radius is now simply set by maximum displacement possible.
    In the future, I might add data-driven approach mentioned in LAP paper (supple pg.7)'''
    dist[dist > DISPLACEMENT] = np.Inf  # assign a large cost for unlikely a pair
    # dist[abs(massdiff) > MASSTHRES] = np.Inf
    cost = dist
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        return labels0, labels

    # Define initial costBorn and costDie in the first frame
    if not hasattr(holder, 'cost_born') or not hasattr(holder, 'cost_die'):
        holder.cost_born = np.percentile(cost[~np.isinf(cost)], 80)
        holder.cost_die = np.percentile(cost[~np.isinf(cost)], 80)
    # try:
    binary_cost = call_lap(cost, holder.cost_die, holder.cost_born)
    # The first assignment of np.Inf is to reduce calculation of linear assignment.
    # This part will make sure that cells outside of these range do not get connected.
    binary_cost[(np.abs(massdiff) > MASSTHRES)] = False
    binary_cost[(dist > DISPLACEMENT)] = False
    gp, gc = np.where(binary_cost)
    idx0, idx1 = list(gp), list(gc)

    for i0, i1 in zip(idx0, idx1):
        labels[labels1 == rps1[i1].label] = rps0[i0].label
        labels0[labels0 == rps0[i0].label] = -rps0[i0].label

    # update cost
    linked_dist = [dist[i0, i1] for i0, i1 in zip(idx0, idx1)]
    if linked_dist:
        cost = np.max(linked_dist)*1.05
        if cost != 0:  # solver freezes if cost is 0
            holder.cost_born, holder.cost_die = cost, cost
    return labels0, labels


def track_neck_cut(img0, img1, labels0, labels1, DISPLACEMENT=10, MASSTHRES=0.2,
                   EDGELEN=5, THRES_ANGLE=180, WSLIMIT=False, SMALL_RAD=None, CANDS_LIMIT=300):
    """
    Adaptive segmentation by using tracking informaiton.
    Separate two objects by making a cut at the deflection. For each points on the outline,
    it will make a triangle separated by EDGELEN and calculates the angle facing inside of concave.

    The majority of cells need to be tracked before the this method to calculate LARGE_RAD and SMALL_RAD.

    EDGELEN (int):      A length of edges of triangle on the nuclear perimeter.
    THRES_ANGLE (int):  Define the neck points if a triangle has more than this angle.
    STEPLIM (int):      points of neck needs to be separated by at least STEPLIM in parimeters.
    WSLIMIT (bool):     Limit search points to ones overlapped with watershed transformed images. Set it True if calculation is slow.

    SMALL_RAD (int or None): The smallest radius of candidate objects. If you have many cells, set it to None will infer the radius from previous frame.
    CANDS_LIMIT(int): use lower if slow. limit a number of searches.

    """
    print('track_neck_cut')
    labels0, labels = nn_closer(img0, img1, labels0, labels1, DISPLACEMENT, MASSTHRES)
    labels1 = -labels.copy()

    if SMALL_RAD is None and not hasattr(holder, 'SMALL_RAD'):
        tracked_area = [i.area for i in regionprops(labels)]
        holder.SMALL_RAD = np.sqrt(np.percentile(tracked_area, 5)/np.pi)
    elif SMALL_RAD is not None:
        holder.SMALL_RAD = SMALL_RAD
    SMALL_RAD = holder.SMALL_RAD

    rps0 = regionprops(labels0, img0)
    unique_labels = np.unique(labels1)
    unique_labels = unique_labels[unique_labels > 0]

    if WSLIMIT:
        wlines = wshed_raw(labels1 > 0, img1)
    else:
        wlines = np.ones(labels1.shape, np.bool)

    store = []
    coords_store = []
    for label_id in unique_labels:
        if label_id == 0:
            continue
        cc = CellCutter(labels1 == label_id, img1, wlines, small_rad=SMALL_RAD,
                        EDGELEN=EDGELEN, THRES=THRES_ANGLE, CANDS_LIMIT=CANDS_LIMIT)
        cc.prepare_coords_set()
        candidates = cc.search_cut_candidates(cc.bw.copy(), cc.coords_set[:CANDS_LIMIT])
        for c in candidates:
            c.raw_label = label_id
        store.append(candidates)
        coords_store.append(cc.coords_set)

    coords_store = [i for i in coords_store if i]
    # Attempt a first cut.
    good_cells = _find_best_neck_cut(rps0, store, DISPLACEMENT, MASSTHRES)
    labels0, labels = _update_labels_neck_cut(labels0, labels1, good_cells)
    labels0, labels = nn_closer(img0, img1, labels0, -labels, DISPLACEMENT, MASSTHRES)

    while good_cells:
        rps0 = regionprops(labels0, img0)
        labels1 = -labels.copy()
        rps0 = regionprops(labels0, img0)
        unique_labels = np.unique(labels1)

        store = []
        for label_id in unique_labels:
            if label_id == 0:
                continue
            bw = labels1 == label_id
            coords_set = [i for i in coords_store if bw[i[0][0][0], i[0][0][1]]]
            if not coords_set:
                continue
            coords_set = coords_set[0]
            candidates = cc.search_cut_candidates(bw, coords_set)
            for c in candidates:
                c.raw_label = label_id
            store.append(candidates)
            coords_store.append(coords_set)
        good_cells = _find_best_neck_cut(rps0, store, DISPLACEMENT, MASSTHRES)
        labels0, labels = _update_labels_neck_cut(labels0, labels1, good_cells)
        labels0, labels = nn_closer(img0, img1, labels0, -labels, DISPLACEMENT, MASSTHRES)
    return labels0, labels



def watershed_distance(img0, img1, labels0, labels1, DISPLACEMENT=10,
                       MASSTHRES=0.2, REGMAX=10, MIN_SIZE=50):
    '''
    Adaptive segmentation by using tracking informaiton.
    watershed existing label, meaning make a cut at the deflection.
    After the cuts, objects will be linked if they are within DISPLACEMENT and MASSTHRES.
    If two candidates are found, it will pick a closer one.
    track_neck_cut may be more sensitive but it takes a long time if objects are not smooth.
    Args:
        REGMAX (int):       Watershed seeds will be separated by at least REGMAX.
                            Smaller REGMAX will allow more cuts.
        DISPLACEMENT (int): The maximum distance (in pixel)
        MASSTHRES (float):  The maximum difference of total intensity changes.
                            0.2 means it allows for 20% total intensity changes.
    '''
    labels0, labels = nn_closer(img0, img1, labels0, labels1, DISPLACEMENT, MASSTHRES)

    def _wd(labels0, labels, img0, img1):
        labels1 = -labels.copy()
        rps0 = regionprops(labels0, img0)

        from subdetect_operation import watershed_divide  # DO NOT MOVE IT
        from utils.track_utils import _find_match
        untracked_labels = labels1.copy()
        untracked_labels[untracked_labels < 0] = 0
        wshed_labels = watershed_divide(untracked_labels, regmax=REGMAX, min_size=MIN_SIZE)
        wshed_labels = label(wshed_labels)

        store = regionprops(wshed_labels, img1)
        good_cells = _find_match(rps0, store, DISPLACEMENT, MASSTHRES)
        for gc in good_cells:  # needed to reuse _update_labels_neck_cut
            gccrds = gc.coords[0]
            gc.raw_label = labels1[gccrds[0], gccrds[1]]
        labels0, labels = _update_labels_neck_cut(labels0, labels1, good_cells)
        labels0, labels = nn_closer(img0, img1, labels0, -labels, DISPLACEMENT, MASSTHRES)
        return labels0, labels, good_cells
    labels0, labels, good_cells = _wd(labels0, labels, img0, img1)
    while good_cells:
        labels0, labels, good_cells = _wd(labels0, labels, img0, img1)
    return labels0, labels
