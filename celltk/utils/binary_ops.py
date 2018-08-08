import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import warnings


def validate_tuple(value, ndim):
    if not hasattr(value, '__iter__'):
        return (value,) * ndim
    if len(value) == ndim:
        return tuple(value)
    raise ValueError("List length should have same length as image dimensions.")


def drop_close(pos, separation, intensity=None):
    """
    Adapted from trackpy
    """
    """ Removes features that are closer than separation from other features.
    When intensity is given, the one with the lowest intensity is dropped:
    else the most topleft is dropped (to avoid randomness)"""
    to_drop = where_close(pos, separation, intensity)
    return np.delete(pos, to_drop, axis=0)


def where_close(pos, separation, intensity=None):
    """
    Adapted from trackpy
    """
    """ Returns indices of features that are closer than separation from other
    features. When intensity is given, the one with the lowest intensity is
    returned: else the most topleft is returned (to avoid randomness)"""
    if len(pos) == 0:
        return []
    separation = validate_tuple(separation, pos.shape[1])
    if any([s == 0 for s in separation]):
        return []
    # Rescale positions, so that pairs are identified below a distance
    # of 1.
    pos_rescaled = pos / separation
    duplicates = cKDTree(pos_rescaled, 30).query_pairs(1 - 1e-7)
    if len(duplicates) == 0:
        return []
    index_0 = np.fromiter((x[0] for x in duplicates), dtype=int)
    index_1 = np.fromiter((x[1] for x in duplicates), dtype=int)
    if intensity is None:
        to_drop = np.where(np.sum(pos_rescaled[index_0], 1) >
                           np.sum(pos_rescaled[index_1], 1),
                           index_1, index_0)
    else:
        intensity = np.asarray(intensity)
        intensity_0 = intensity[index_0]
        intensity_1 = intensity[index_1]
        to_drop = np.where(intensity_0 > intensity_1, index_1, index_0)
        edge_cases = intensity_0 == intensity_1
        if np.any(edge_cases):
            index_0 = index_0[edge_cases]
            index_1 = index_1[edge_cases]
            to_drop[edge_cases] = np.where(np.sum(pos_rescaled[index_0], 1) >
                                           np.sum(pos_rescaled[index_1], 1),
                                           index_1, index_0)
    return np.unique(to_drop)


def grey_dilation(image, separation, percentile=64, margin=None, precise=True):
    """
    Adapted and modified from trackpy. It takes images not integer types.
    """
    """Find local maxima whose brightness is above a given percentile.
    Parameters
    ----------
    image : ndarray
        For best performance, provide an integer-type array. If the type is not
        of integer-type, the image will be normalized and coerced to uint8.
    separation : number or tuple of numbers
        Minimum separation between maxima. See precise for more information.
    percentile : float in range of [0,100], optional
        Features must have a peak brighter than pixels in this percentile.
        This helps eliminate spurious peaks. Default 64.
    margin : integer or tuple of integers, optional
        Zone of exclusion at edges of image. Default is ``separation / 2``.
    precise : boolean, optional
        Determines whether there will be an extra filtering step (``drop_close``)
        discarding features that are too close. Degrades performance.
        Because of the square kernel used, too many features are returned when
        precise=False. Default True.
    See Also
    --------
    drop_close : removes features that are too close to brighter features
    grey_dilation_legacy : local maxima finding routine used until trackpy v0.3
    """

    ndim = image.ndim
    separation = validate_tuple(separation, ndim)
    if margin is None:
        margin = tuple([int(s / 2) for s in separation])

    # Compute a threshold based on percentile.
    threshold = np.percentile(image, percentile)
    if np.isnan(threshold):
        warnings.warn("Image is completely black.", UserWarning)
        return np.empty((0, ndim))

    # Find the largest box that fits inside the ellipse given by separation
    size = [int(2 * s / np.sqrt(ndim)) for s in separation]

    # The intersection of the image with its dilation gives local maxima.
    dilation = ndimage.grey_dilation(image, size, mode='constant')
    maxima = (image == dilation) & (image > threshold)
    if np.sum(maxima) == 0:
        warnings.warn("Image contains no local maxima.", UserWarning)
        return np.empty((0, ndim))

    pos = np.vstack(np.where(maxima)).T

    # Do not accept peaks near the edges.
    shape = np.array(image.shape)
    near_edge = np.any((pos < margin) | (pos > (shape - margin - 1)), 1)
    pos = pos[~near_edge]

    if len(pos) == 0:
        warnings.warn("All local maxima were in the margins.", UserWarning)
        return np.empty((0, ndim))

    # Remove local maxima that are too close to each other
    if precise:
        pos = drop_close(pos, separation, image[maxima][~near_edge])
    return pos
