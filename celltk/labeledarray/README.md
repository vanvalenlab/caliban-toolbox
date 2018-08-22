# LabeledArray

Numpy array subclass for indexing by strings.  
  
Using multi-index in pandas sometimes provides complications in terms of "copies vs views". This array is to provide numpy.array's behavior and still enable to slice array by strings.

Underlying data can be 2D, 3D or N-dimensional array. First dimension will be used for labels (multi-index).

```
arr = np.zeros((3, 20, 100))
labels = np.array([['nuc' ,'area', ''],
                   ['nuc' ,'FITC' , 'min_intensity'],
                   ['nuc' ,'FITC' , 'max_intensity']], dtype=object)
larr = LabeledArray(arr, labels)
print larr.shape
print larr['nuc', 'FITC'].shape
print larr['nuc', 'FITC', 'max_intensity'].shape
```

The extra attributes including labels are automatically saved and loaded with the array. 
```
larr = LabeledArray(arr, labels)
larr.time = np.arange(arr.shape[-1])
larr.save('temp')
new_larr = LabeledArray().load('temp')
print new_larr.time
```
