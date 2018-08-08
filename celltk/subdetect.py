"""
python celltk/subdetect.py -l c3/img_00000000* -f ring_dilation -o c4 -p MARGIN=0
"""

from utils.util import imread
import tifffile as tiff
import argparse
from os.path import basename, join
import numpy as np
import subdetect_operation
from itertools import izip_longest
from utils.file_io import make_dirs, imsave, lbread
from utils.parser import ParamParser
from utils.global_holder import holder
import logging

logger = logging.getLogger(__name__)


def caller(inputs, inputs_labels, output, functions, params):
    make_dirs(output)

    logger.info("Functions {0} for {1} images.".format(functions, len(inputs)))
    img = None
    for holder.frame, (path, pathl) in enumerate(izip_longest(inputs, inputs_labels)):
        if path is not None:
            img = imread(path)
        labels0 = lbread(pathl)
        for function, param in zip(functions, params):
            func = getattr(subdetect_operation, function)
            if img is not None:
                labels = func(labels0, img, **param)
            else:
                labels = func(labels0, **param)
        imsave(labels, output, pathl, dtype=np.int16)
        logger.info("\tframe {0} done.".format(holder.frame))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="images", nargs="*", default=[])
    parser.add_argument("-l", "--labels", help="labels", nargs="+", default=[])
    parser.add_argument("-o", "--output", help="output directory",
                        type=str, default='temp')
    parser.add_argument("-f", "--functions", help="functions", nargs="+")
    parser.add_argument('-p', '--param', nargs='+', help='parameters', action='append')
    args = parser.parse_args()

    if args.functions is None:
        print help(subdetect_operation)
        return

    params = ParamParser(args.param).run()
    holder.args = args

    caller(args.input, args.labels, args.output, args.functions, params)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
