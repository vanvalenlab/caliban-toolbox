from skimage.measure._regionprops import _RegionProperties
import numpy as np
import scipy.ndimage as ndi
from global_holder import holder
from math import sqrt
from filters import label


class _RegionProperties2(_RegionProperties):
    parent = None
    nxt = None

    @property
    def total_intensity(self):
        return np.sum(self.intensity_image[self.image])

    @property
    def x(self):
        return self.centroid[1]

    @property
    def y(self):
        return self.centroid[0]

    @property
    def median_intensity(self):
        return np.median(self.intensity_image[self.image])

    @property
    def std_intensity(self):
        return np.std(self.intensity_image[self.image])

    @property
    def cv_intensity(self):
        return self.std_intensity/self.mean_intensity

    @property
    def cell_id(self):
        return self.label

    @property
    def minor_axis_length(self):
        _, l2 = self.inertia_tensor_eigvals
        return 4 * sqrt(max(l2, 0))

    @property
    def major_axis_length(self):
        l1, _ = self.inertia_tensor_eigvals
        return 4 * sqrt(max(l1, 0))

    @property
    def num_seg(self):
        """
        number of isolated segments
        """
        return np.max(label(self.image, connectivity=1))

def regionprops(label_image, intensity_image=None, cache=True):
    label_image = np.squeeze(label_image)

    if label_image.ndim not in (2, 3):
        raise TypeError('Only 2-D and 3-D images supported.')

    if not np.issubdtype(label_image.dtype, np.integer):
        raise TypeError('Label image must be of integral type.')

    regions = []

    objects = ndi.find_objects(label_image)
    for i, sl in enumerate(objects):
        if sl is None:
            continue

        label = i + 1

        props = _RegionProperties2(sl, label, label_image, intensity_image, cache)
        regions.append(props)

    return regions


class Cell(object):
    def __init__(self, regionprop):
        for p in dir(regionprop):
            if "__" in p:
                continue
            ret = getattr(regionprop, p)
            if not isinstance(ret, dict) and not isinstance(ret, tuple) and not isinstance(ret, np.ndarray):
                setattr(self, p, ret)
        self.coords = regionprop.coords
        self.abs_id = holder.count()
        self.centroid = regionprop.centroid


class LCell(object):
    def __init__(self, regionprop):
        for p in ['area', 'num_seg', 'median_intensity', 'total_intensity', 'x', 'y', 'cell_id', 'label']:
            setattr(self, p, getattr(regionprop, p))
        self.abs_id = holder.count()
        self._original_label = self.label
        self.nxt = None
        self.parent = None

# class LCell(object):
#     def __init__(self, regionprop):
#         self._labels = ['area', 'num_seg', 'median_intensity', 'total_intensity', 'x', 'y', 'cell_id', 'label']
#         self._data = np.zeros(len(self._labels))
#         for num, p in enumerate(self._labels):
#             self._data[num] = getattr(regionprop, p)
#         self.parent = None
#         self.nxt =None
    
#     def __getattr__(self, name):
#         if name in self._labels:
#             return self._data[self._labels.index(name)]
#         super(LCell, self).__getattr__(name)

#     def __setattr__(self, name, value):
#         object.__setattr__(self, name, value)
#         if name in self._labels:
#             self._data[self._labels.index(name)] = value
