import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import watershed as skiwatershed
from skimage.measure import label
from skimage.feature import peak_local_max


def dilate_sitk(labels, RAD):
    slabels = sitk.GetImageFromArray(labels)
    gd = sitk.GrayscaleDilateImageFilter()
    gd.SetKernelRadius(RAD)
    return sitk.GetArrayFromImage(gd.Execute(slabels))


def voronoi_expand(labels, return_line=False):
    dist = distance_transform_edt(labels)

    vor = skiwatershed(-dist, markers=labels)
    if not return_line:
        return vor
    else:
        mask = skiwatershed(-dist, markers=labels, watershed_line=True)
        lines = mask == 0
        return vor, lines


def calc_mask_exclude_overlap(nuclabel, RINGWIDTH=5):
    """
    Examples:
        >>> template = np.zeros((5, 5))
        >>> template[1, 1] = 1
        >>> template[-2, -2] = 2
        >>> calc_mask_exclude_overlap(template, 2)
        array([[False, False, False, False, False],
               [False, False,  True,  True, False],
               [False,  True,  True,  True, False],
               [False,  True,  True, False, False],
               [False, False, False, False, False]], dtype=bool)
    """
    dilated_nuc = dilate_sitk(nuclabel.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 6e4 - nuclabel
    comp_dilated_nuc[comp_dilated_nuc == 6e4] = 0
    comp_dilated_nuc = dilate_sitk(comp_dilated_nuc.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 6e4 - comp_dilated_nuc
    comp_dilated_nuc[comp_dilated_nuc == 6e4] = 0
    mask = comp_dilated_nuc != dilated_nuc
    return mask


def dilate_to_cytoring(labels, RINGWIDTH, MARGIN):
    """
    Examples:
        >>> template = np.zeros((5, 5))
        >>> template[2, 2] = 1
        >>> dilate_to_cytoring(template, 1, 0)
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]], dtype=uint16)
        >>> dilate_to_cytoring(template, 2, 1)
        array([[0, 1, 1, 1, 0],
               [1, 0, 0, 0, 1],
               [1, 0, 0, 0, 1],
               [1, 0, 0, 0, 1],
               [0, 1, 1, 1, 0]], dtype=uint16)
    """
    dilated_nuc = dilate_sitk(labels.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - labels
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    comp_dilated_nuc = dilate_sitk(comp_dilated_nuc.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - comp_dilated_nuc
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    dilated_nuc[comp_dilated_nuc != dilated_nuc] = 0
    if MARGIN == 0:
        antinucmask = labels
    else:
        antinucmask = dilate_sitk(np.int32(labels), MARGIN)
    dilated_nuc[antinucmask.astype(bool)] = 0
    return dilated_nuc.astype(np.uint16)


def dilate_to_cytoring_buffer(labels, RINGWIDTH, MARGIN, BUFFER):
    """
    Examples:
        >>> template = np.zeros((5, 5))
        >>> template[1, 1] = 1
        >>> template[-2, -2] = 2
        >>> dilate_to_cytoring_buffer(template, 2, 0, 1)
        array([[1, 1, 0, 0, 0],
               [1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 2],
               [0, 0, 0, 2, 2]], dtype=uint16)
    """
    dilated_nuc = dilate_sitk(labels.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - labels
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0
    comp_dilated_nuc = dilate_sitk(comp_dilated_nuc.astype(np.int32), RINGWIDTH)
    comp_dilated_nuc = 1e4 - comp_dilated_nuc
    comp_dilated_nuc[comp_dilated_nuc == 1e4] = 0

    vor, vlines = voronoi_expand(labels, return_line=True)
    vor0, vor1 = vor.copy(), vor.copy()
    vor0[comp_dilated_nuc != vor0.copy()] = False
    vor1[dilated_nuc != vor1.copy()] = False
    dilated_nuc = np.max(np.dstack((vor0, vor1)), axis=2)

    if MARGIN == 0:
        antinucmask = labels
    else:
        antinucmask = dilate_sitk(np.int32(labels), MARGIN)
    dilated_nuc[antinucmask.astype(bool)] = 0

    if BUFFER:
        vlines = dilate_sitk(vlines.astype(np.uint32), BUFFER)
        dilated_nuc[vlines > 0] = 0
    return dilated_nuc.astype(np.uint16)


def watershed_labels(labels, regmax):
    # Since there are non-unique values for dist, add very small numbers. This will separate each marker by regmax at least.
    dist = distance_transform_edt(labels) + np.random.rand(*labels.shape)*1e-10
    labeled_maxima = label(peak_local_max(dist, min_distance=int(regmax), indices=False))
    wshed = -dist
    wshed = wshed - np.min(dist)
    markers = np.zeros(wshed.shape, np.int16)
    markers[labeled_maxima > 0] = -labeled_maxima[labeled_maxima > 0]
    markers[labels == 0] = 0
    wlabels = skiwatershed(wshed, markers, connectivity=np.ones((3,3), bool), mask=labels!=0)
    wlabels = -wlabels
    wlabels = labels.max() + wlabels
    wlabels[wlabels == labels.max()] = 0
    all_labels = label(labels + wlabels)
    return all_labels
