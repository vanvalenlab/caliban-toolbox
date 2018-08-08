"""
python celltk/segment.py -i ~/covertrack/data/testimages/img_0000000* -f example_thres -o c1 -p THRES=2000
"""


# from scipy.ndimage import imread
import argparse
import tifffile as tiff
from os.path import basename, join
import numpy as np
import segment_operation
from skimage.segmentation import clear_border
from utils.filters import gray_fill_holes
from skimage.morphology import remove_small_objects
from utils.filters import label
from scipy.ndimage.morphology import binary_opening
from utils.util import imread
from utils.file_io import make_dirs, imsave
from utils.parser import ParamParser, parse_image_files
from utils.global_holder import holder
import logging

logger = logging.getLogger(__name__)

radius = [3, 50]


def clean_labels(labels, rad, OPEN=2):
    """default cleaning. Fill holes, remove small and large objects and opening.
    """
    labels = gray_fill_holes(labels)
    labels = clear_border(labels, buffer_size=2)
    labels = remove_small_objects(labels, rad[0]**2 * np.pi, connectivity=4)
    antimask = remove_small_objects(labels, rad[1]**2 * np.pi, connectivity=4)
    labels[antimask > 0] = False
    labels = label(binary_opening(labels, np.ones((int(OPEN), int(OPEN))), iterations=1))
    return labels


def caller(inputs, output, functions, params):
    make_dirs(output)
    logger.info("Functions {0} for {1} images.".format(functions, len(inputs)))

    for holder.frame, path in enumerate(inputs):
        holder.path = path
        img = imread(path)
        for function, param in zip(functions, params):
            func = getattr(segment_operation, function)
            img = func(img, **param)
        if isinstance(path, list) or isinstance(path, tuple):
            path = path[0]
        labels = clean_labels(img, radius)
        imsave(labels, output, path, dtype=np.int16)
        logger.info("\tframe {0}: {1} objects segmented.".format(holder.frame, len(np.unique(labels))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="images", nargs="*")
    parser.add_argument("-o", "--output", help="output directory",
                        type=str, default='temp')
    parser.add_argument("-f", "--functions", help="functions", nargs="*", default=None)
    parser.add_argument('-p', '--param', nargs='+', help='parameters', action='append')
    args = parser.parse_args()

    params = ParamParser(args.param).run()
    if args.functions is None:
        print help(segment_operation)
        return

    holder.args = args
    args.input = parse_image_files(args.input)

    caller(args.input, args.output, args.functions, params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
