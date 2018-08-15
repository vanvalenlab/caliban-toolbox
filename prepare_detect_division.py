"""
python celltk/postprocess.py -f gap_closing -i c0/img_00000000* -l c2/img_00000000*  -o c3 -p DISPLACEMENT=30
"""

from scipy.ndimage import imread
from os.path import basename, join
import numpy as np
import detect_division
from celltk.utils.postprocess_utils import regionprops, LCell # set default parent and next as None
from celltk.utils.file_io import make_dirs, imsave, lbread
from celltk.utils.filters import labels2outlines
from scipy.ndimage import binary_dilation
import logging

logger = logging.getLogger(__name__)


def cells2labels(cells, frame, labels):
    inframe = [i for i in cells if i.frame == frame]
    template = np.zeros(labels.shape)
    for cell in inframe:
        template[labels == cell._original_label] = cell.label
        if cell.parent is not None:
            outline = labels2outlines(binary_dilation(template == cell.label))
            # outline = labels2outlines(template == cell.label)
            template[outline > 0] = -cell.parent
    return template


def caller(inputs, inputs_labels, output, functions, params):
    print(inputs, inputs_labels)
    print(output, functions)
    print(params)
    make_dirs(output)

    # Make cells. cells are a list of regionproperties or subclasses.
    logger.info('Postprocess.\tcollecting cells...')
    store = []
    for frame, (path, pathl) in enumerate(zip(inputs, inputs_labels)):
        img, labels = imread(path), lbread(pathl)
        cells = regionprops(labels, img)
        cells = [LCell(cell) for cell in cells]
        for cell in cells:
            cell.frame = frame
            if frame > 0:
                all_labels = [i.label for i in store[frame - 1]]
                if cell.label in all_labels:
                    store[frame - 1][all_labels.index(cell.label)].nxt = cell
        store.append(cells)
    cells = [i for j in store for i in j]
    # Each function receives cells (regionprops) and finally return labels generated by cells.label
    for function, param in zip(functions, params):
        logger.info('\trunning {0}'.format(function))
        func = getattr(detect_division, function)
        print(func)
        cells = func(cells)

    logger.info('\tsaving images...')
    for frame, (path, pathl) in enumerate(zip(inputs, inputs_labels)):
        labels = cells2labels(cells, frame, lbread(pathl))
        imsave(labels, output, path, dtype=np.int16)


def main():
    inputs = ['/data/00_0/raw/set_0_x_0_y_0_slice_00.png', '/data/00_0/raw/set_0_x_0_y_0_slice_01.png', '/data/00_0/raw/set_0_x_0_y_0_slice_02.png', '/data/00_0/raw/set_0_x_0_y_0_slice_03.png', '/data/00_0/raw/set_0_x_0_y_0_slice_04.png', '/data/00_0/raw/set_0_x_0_y_0_slice_05.png', '/data/00_0/raw/set_0_x_0_y_0_slice_06.png', '/data/00_0/raw/set_0_x_0_y_0_slice_07.png', '/data/00_0/raw/set_0_x_0_y_0_slice_08.png', '/data/00_0/raw/set_0_x_0_y_0_slice_09.png', '/data/00_0/raw/set_0_x_0_y_0_slice_10.png', '/data/00_0/raw/set_0_x_0_y_0_slice_11.png', '/data/00_0/raw/set_0_x_0_y_0_slice_12.png', '/data/00_0/raw/set_0_x_0_y_0_slice_13.png', '/data/00_0/raw/set_0_x_0_y_0_slice_14.png', '/data/00_0/raw/set_0_x_0_y_0_slice_15.png', '/data/00_0/raw/set_0_x_0_y_0_slice_16.png', '/data/00_0/raw/set_0_x_0_y_0_slice_17.png', '/data/00_0/raw/set_0_x_0_y_0_slice_18.png', '/data/00_0/raw/set_0_x_0_y_0_slice_19.png', '/data/00_0/raw/set_0_x_0_y_0_slice_20.png', '/data/00_0/raw/set_0_x_0_y_0_slice_21.png', '/data/00_0/raw/set_0_x_0_y_0_slice_22.png', '/data/00_0/raw/set_0_x_0_y_0_slice_23.png', '/data/00_0/raw/set_0_x_0_y_0_slice_24.png', '/data/00_0/raw/set_0_x_0_y_0_slice_25.png', '/data/00_0/raw/set_0_x_0_y_0_slice_26.png', '/data/00_0/raw/set_0_x_0_y_0_slice_27.png', '/data/00_0/raw/set_0_x_0_y_0_slice_28.png', '/data/00_0/raw/set_0_x_0_y_0_slice_29.png', '/data/00_0/raw/set_0_x_0_y_0_slice_30.png', '/data/00_0/raw/set_0_x_0_y_0_slice_31.png', '/data/00_0/raw/set_0_x_0_y_0_slice_32.png', '/data/00_0/raw/set_0_x_0_y_0_slice_33.png', '/data/00_0/raw/set_0_x_0_y_0_slice_34.png', '/data/00_0/raw/set_0_x_0_y_0_slice_35.png', '/data/00_0/raw/set_0_x_0_y_0_slice_36.png', '/data/00_0/raw/set_0_x_0_y_0_slice_37.png', '/data/00_0/raw/set_0_x_0_y_0_slice_38.png', '/data/00_0/raw/set_0_x_0_y_0_slice_39.png']

    inputs_labels = ['/data/00_0/annotated/00.tif', '/data/00_0/annotated/01.tif', '/data/00_0/annotated/02.tif', '/data/00_0/annotated/03.tif', '/data/00_0/annotated/04.tif', '/data/00_0/annotated/05.tif', '/data/00_0/annotated/06.tif', '/data/00_0/annotated/07.tif', '/data/00_0/annotated/08.tif', '/data/00_0/annotated/09.tif', '/data/00_0/annotated/10.tif', '/data/00_0/annotated/11.tif', '/data/00_0/annotated/12.tif', '/data/00_0/annotated/13.tif', '/data/00_0/annotated/14.tif', '/data/00_0/annotated/15.tif', '/data/00_0/annotated/16.tif', '/data/00_0/annotated/17.tif', '/data/00_0/annotated/18.tif', '/data/00_0/annotated/19.tif', '/data/00_0/annotated/20.tif', '/data/00_0/annotated/21.tif', '/data/00_0/annotated/22.tif', '/data/00_0/annotated/23.tif', '/data/00_0/annotated/24.tif', '/data/00_0/annotated/25.tif', '/data/00_0/annotated/26.tif', '/data/00_0/annotated/27.tif', '/data/00_0/annotated/28.tif', '/data/00_0/annotated/29.tif', '/data/00_0/annotated/30.tif', '/data/00_0/annotated/31.tif', '/data/00_0/annotated/32.tif', '/data/00_0/annotated/33.tif', '/data/00_0/annotated/34.tif', '/data/00_0/annotated/35.tif', '/data/00_0/annotated/36.tif', '/data/00_0/annotated/37.tif', '/data/00_0/annotated/38.tif', '/data/00_0/annotated/39.tif']

    output = '/home/set0output/00_0/nuc'
    functions = ['detect_division']
    params = {'DIVISIONMASSERR': 0.35, 'DISPLACEMENT': 50}
    caller(inputs, inputs_labels, output, functions, params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
