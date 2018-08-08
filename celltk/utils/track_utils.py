from __future__ import division
import numpy as np
from munkres import munkres
from scipy.spatial.distance import cdist


def calc_ratiodiff(a, b):
    """calculate how much pairwise ratio of vector a to b.
    """
    a0, a1 = np.meshgrid(a, b)
    return (a1.T - a0.T)/a0.T


def calc_diff(a, b):
    """calculate how much pairwise ratio of vector a to b.
    """
    a0, a1 = np.meshgrid(a, b)
    return a1.T - a0.T


def calc_massdiff(cells0, cells1):
    return calc_ratiodiff([i.total_intensity for i in cells0], [i.total_intensity for i in cells1])


def find_one_to_one_assign(cost):
    (_, col1) = np.where([np.sum(cost, 0) == 1])
    cost[np.sum(cost, 1) != 1] = False
    (row, col2) = np.where(cost)
    good_curr_idx = [ci for ri, ci in zip(row, col2) if ci in col1]
    good_prev_idx = [ri for ri, ci in zip(row, col2) if ci in good_curr_idx]
    return good_curr_idx, good_prev_idx


def prepare_costmat(cost, costDie, costBorn):
    '''d is cost matrix,
    often distance matrix where rows are previous and columns as current
    d contains NaN where tracking of those two objects are not possible.
    costDie and costBorn'''
    cost[np.isnan(cost)] = np.Inf  # give a large cost.
    costDieMat = np.array(np.float64(np.diag([costDie]*cost.shape[0])))  # diagonal
    costBornMat = np.array(np.float64(np.diag([costBorn]*cost.shape[1])))
    costDieMat[costDieMat == 0] = np.Inf
    costBornMat[costBornMat == 0] = np.Inf

    costMat = np.ones((sum(cost.shape), sum(cost.shape)))*np.Inf
    costMat[0:cost.shape[0], 0:cost.shape[1]] = cost
    costMat[-cost.shape[1]:, 0:cost.shape[1]] = costBornMat
    costMat[0:cost.shape[0], -cost.shape[0]:] = costDieMat
    lowerRightBlock = cost.transpose()
    costMat[cost.shape[0]:, cost.shape[1]:] = lowerRightBlock
    return costMat


def call_lap(cost, costDie, costBorn):
    costMat = prepare_costmat(cost, costDie, costBorn)
    t = munkres(costMat)
    topleft = t[0:cost.shape[0], 0:cost.shape[1]]
    return topleft


def pick_closer_binarycostmat(binarymat, distmat):
    '''
    pick closer cells if there are two similar nucleus within area
    '''
    twonuc = np.where(np.sum(binarymat, 1) == 2)[0]
    for ti in twonuc:
        di = distmat[ti, :]
        bi = binarymat[ti, :]
        binarymat[ti, :] = min(di[bi]) == di
    return binarymat


def pick_closer_cost(binarymat, distmat):
    distmat = distmat.astype(np.float)
    bmat = np.zeros(binarymat.shape, np.bool)
    distmat[~binarymat] = np.Inf
    true_rows = np.unique(np.where(binarymat)[0])
    for i in true_rows:
        bmat[i, np.argmin(distmat[i, :])] = True
    return bmat


def _find_best_neck_cut(rps0, store, DISPLACEMENT, MASSTHRES):
    """called by track_neck_cut only
    """
    good_cells = []
    for cands in store:
        if not cands or not rps0:
            continue
        dist = cdist([i.centroid for i in rps0], [i.centroid for i in cands])
        massdiff = calc_massdiff(rps0, cands)
        binary_cost = (dist < DISPLACEMENT) * (abs(massdiff) < MASSTHRES)
        line_int = [i.line_total for i in cands]
        line_mat = np.tile(line_int, len(rps0)).reshape(len(rps0), len(line_int))
        binary_cost = pick_closer_cost(binary_cost, line_mat)
        binary_cost = pick_closer_cost(binary_cost.T, dist.T).T
        idx1, idx0 = find_one_to_one_assign(binary_cost.copy())
        if not idx0:
            continue
        i0, i1 = idx0[0], idx1[0]
        cell = cands[i1]
        cell.previous = rps0[i0]
        good_cells.append(cell)
    return good_cells


def _update_labels_neck_cut(labels0, labels1, good_cells):
    """called by track_neck_cut only
    """
    labels = -labels1.copy()
    minint = -np.max(abs(labels))
    unique_raw_labels = np.unique([cell.raw_label for cell in good_cells])
    neg_labels = np.zeros(labels.shape, np.int32)
    for i in unique_raw_labels:
        minint -= 1
        neg_labels[labels1 == i] = minint
        labels[labels1 == i] = 0

    for cell in good_cells:
        for c0, c1 in cell.coords:
            neg_labels[c0, c1] = cell.previous.label
        labels0[labels0 == cell.previous.label] = -cell.previous.label
    labels = labels + neg_labels
    return labels0, labels


def _find_match(rps0, cands, DISPLACEMENT, MASSTHRES):
    good_cells = []
    if not cands or not rps0:
        return []
    dist = cdist([i.centroid for i in rps0], [i.centroid for i in cands])
    massdiff = calc_massdiff(rps0, cands)
    binary_cost = (dist < DISPLACEMENT) * (abs(massdiff) < MASSTHRES)
    binary_cost = pick_closer_cost(binary_cost, dist)
    binary_cost = pick_closer_cost(binary_cost.T, dist.T).T
    idx1, idx0 = find_one_to_one_assign(binary_cost.copy())
    if not idx0:
        return []
    i0, i1 = idx0[0], idx1[0]
    cell = cands[i1]
    cell.previous = rps0[i0]
    good_cells.append(cell)
    return good_cells

