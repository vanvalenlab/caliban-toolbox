"""
TODO: check if labels is unique.

"""
import numpy as np
from collections import OrderedDict
from utils import sort_labels_and_arr, uniform_list_length


class LabeledArray(np.ndarray):
    """
    Each rows corresponds to labels, each columns corresponds to cells.
    Underlying data structure can be N-dimensional array. First dimension will be used for labeled array.

    Examples:
        >> arr = np.arange(12).reshape((3, 2, 2))
        >> labelarr = np.array([['a1' ,'b1', ''], 
                                ['a1' ,'b2' , 'c1'], 
                                ['a1' ,'b2' , 'c2']], dtype=object)
        >> darr = DArray(arr, labelarr)
        >> assert darr['a1'].shape
        (3, 2, 2)
        >> darr['a1', 'b1'].shape
        (2, 2)
        >> darr['a1', 'b2', 'c1']
        DArray([[4, 5],
               [6, 7]])
    """

    idx = None
    labels = None

    def __new__(cls, arr=None, labels=None, idx=None):
        if arr is None:
            return np.asarray(arr).view(cls)
        labels, arr = sort_labels_and_arr(labels, arr)
        if not isinstance(labels, np.ndarray) and labels is not None:
            labels = np.array(uniform_list_length(labels), dtype=object)
        obj = np.asarray(arr).view(cls)
        obj.labels = labels
        obj.idx = idx
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.labels = getattr(obj, 'labels', None)
        if self.labels is None: return
        if hasattr(obj, 'idx') and self.ndim >= 1:
            if obj.idx is None: return
            if isinstance(obj.idx, int):
                self.labels = self.labels[obj.idx]
            else:
                self.labels = self.labels[obj.idx[0]]
            if isinstance(self.labels, str):
                return
            if self.labels.ndim > 1:
                f_leftshift = lambda a1:all(x>=y for x, y in zip(a1, a1[1:]))
                all_column = np.all(self.labels == self.labels[0,:], axis=0)
                sl = 0 if not f_leftshift(all_column) else all_column.sum()
                self.labels = self.labels[:, slice(sl, None)]
            if self.labels.ndim == 1:
                self.labels = None
                obj = np.array(obj)

    def __getitem__(self, item):
        if isinstance(item, str):
            item = self._label2idx(item)
        if isinstance(item, tuple):
            if isinstance(item[0], str):
                item = self._label2idx(item)
        self.idx = item
        ret = super(LabeledArray, self).__getitem__(item)
        return ret.squeeze()

    def _label2idx(self, item):
        item = (item, ) if not isinstance(item, tuple) else item
        boolarr = np.ones(self.labels.shape[0], dtype=bool)
        for num, it in enumerate(item):
            boolarr = boolarr * (self.labels[:, num]==it)
        tidx = np.where(boolarr)[0]
        if boolarr.sum() == 1:
            return tuple(tidx)
        if boolarr.all():
            return (slice(None, None, None), ) + (slice(None, None, None), ) * (self.ndim - 1)
        minidx = min(tidx) if min(tidx) > 0 else None
        maxidx = max(tidx)+1 if max(tidx)+1 < self.shape[0] else None
        if boolarr.sum() > 1:
            return (slice(minidx, maxidx, None), ) + (slice(None, None, None), ) * (self.ndim - 1)

    def vstack(self, larr):
        """merging first dimension (more labels)
        """
        if self.ndim > larr.ndim:
            larr = np.expand_dims(larr, axis=0)
        return LabeledArray(np.vstack((self, larr)), np.vstack((self.labels, larr.labels)))

    def hstack(self, larr):
        """merging second dimension (more cells)
        """
        if (self.labels == larr.labels).all():
            return LabeledArray(np.hstack((self, larr)), self.labels)

    def save(self, file_name):
        extra_fields = set(dir(self)).difference(set(dir(self.__class__)))
        data = dict(arr=self, labels=self.labels)
        for ef in extra_fields:
            data[ef] = getattr(self, ef)
        np.savez_compressed(file_name, **data)

    @classmethod
    def load(cls, file_name):
        if not file_name.endswith('.npz'):
            file_name = file_name + '.npz'
        f = np.load(file_name)
        arr, labels = f['arr'], f['labels']
        la = LabeledArray(arr, labels)
        for key, value in f.iteritems():
            if not ('arr' == key or 'labels' == key):
                setattr(la, key, value)
        return la


if __name__ == "__main__":
    # Check 2D.
    arr = np.random.rand(3, 100)
    labelarr = np.array([['a1', 'b1', ''], 
                        ['a1' ,'b2' , 'c1'], 
                        ['a1' ,'b2' , 'c2']], dtype=object)
    darr = LabeledArray(arr, labelarr)
    # stop
    assert darr['a1'].shape == (3, 100)
    assert darr['a1', 'b1'].shape == (100, )
    assert darr['a1', 'b2'].shape == (2, 100)
    assert darr['a1', 'b2', 'c1'].shape == (100, )

    # check 3D.
    arr = np.arange(12).reshape((3, 2, 2))
    labelarr = np.array([['a1' ,'b1', ''], 
                        ['a1' ,'b2' , 'c1'], 
                        ['a1' ,'b2' , 'c2']], dtype=object)
    darr = LabeledArray(arr, labelarr)
    assert darr['a1'].shape == (3, 2, 2)
    assert darr['a1', 'b1'].shape == (2, 2)
    assert darr['a1', 'b2'].shape == (2, 2, 2)
    assert darr['a1', 'b2', 'c1'].shape == (2, 2)
    assert darr.shape == (3, 2, 2)
    assert darr[1:, :, :].shape == (2, 2, 2)
    assert darr[1, :, :].shape == (2, 2)
    assert np.all(darr['a1', 'b2'].labels == np.array([['c1'], ['c2']]))

    # can save and load extra fields. add "time" for example.
    darr.time = np.arange(darr.shape[-1])
    darr.save('test')
    cc = LabeledArray().load('test.npz')
    assert cc.time.shape == (2,)
    cc[0:2, :, :]
    cc['a1', 'b1'][0, 0] = 100
    assert np.sum(cc == 100) == 1

    assert darr.vstack(darr).shape == (2 * darr.shape[0], darr.shape[1], darr.shape[2])
    assert darr.hstack(darr).shape == (darr.shape[0], 2 * darr.shape[1], darr.shape[2])


