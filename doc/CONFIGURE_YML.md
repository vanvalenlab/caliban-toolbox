Caller Example:
```
python celltk/caller.py input_files/ktr_inputs/input_anisoinh.yml
```
This yaml file contains the following items:
- OUTPUT_DIR: a parental folder to save output
- operations: a sequence of operations implemented in the order.


We call a set of function name, input, output and parameters as an operation.  
Here is the example of the operation:
```
- function: flatfield_references
  images: /home/KTRimages/IL1B/Pos006/*DAPI*
  output: op000
  params:
    ff_paths: /home/KTRimages/IL1B/FF/*DAPI*
```
Each operation contains ___function___, ___images (and/or labels)___ , ___output___ and ___params___.  


### function and params
The ___celltk\/\*\_operations.py___ modules contains a list of functions, which they take an input image and transform it.  
___function___ is simply a function name from these modules. You can find a function called ___flatfield\_refernces___ in ___celltk/preprocess\_operations.py___; ___ff_paths___ is an argument for this function.

You can quickly check available functions by typing one of the following commands:
```
celltk-preprocess
celltk-segment
celltk-subdetect
celltk-track
celltk-postprocess
```


### images (and/or labels) and output

#### Lazy syntax  
For specifying a series of image files, you may use file names specified by wildcards or a folder containing the files (either relative or absolute). Whenever you use the relative path, it will look for a path under OUTPUT_DIR.   
For example, imagine you have two files as _/home/example/op/img0.tif_ and _/home/example/op/img1.tif_. If `OUTPUT_DIR: /home/example/`, then any of the following syntaxes can be used to take these images as input:
```
images: /home/example/op/img*.tif
images: /home/example/op
images: op/img*.tif
images: op
```

#### A sequence of a sequence
Operations can also be specified as a nested sequence instead of a sequence. 

```
- - function: example0
    images: op0
  - function: example1
  - function: example2
    output: op1
```

In this case, it takes a series of images from op0, apply functions  example0 to example2, and finally save the transformed images in op1 folder. Note that images and output are specified only in the first and last operation, respectively.  

Typically, tracking uses this format so that it enables an incremental tracking with several algorithms. Objects which could not be tracked by the first algorithm are passed to the second algorithm, and so on.  
Tracking e.g.
```
- - function: run_lap
    images: DAPI
    labels: op001
    params:
      MASSTHRES: 0.25
  - function: track_neck_cut
    params:
      THRES_ANGLE: 180
  - function: track_neck_cut
    params:
      THRES_ANGLE: 160
    output: tracked
```
In this example, cells are first attempted for tracking with ___run\_lap___; cells that were not linked by ___run\_lap___ is then passed to ___track\_neck\_cut___ with _THRES\_ANGLE: 180_, and whatever left is finally passed to ___track\_neck\_cut___ with _THRES\_ANGLE: 160_.  
In general, order tracking algorithms from conservative to more radical algorithms.

At the end, however, we want to extract vectors or arrays representing extracted parameters. This is implemented through apply.


