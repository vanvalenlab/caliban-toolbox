"""
python celltk/tracking.py -f nearest_neighbor -i c0/img_00000000* -l c1/img_0000000
0*  -p DISPLACEMENT=10 MASSTHRES=0.2
"""

from utils.util import imread
import argparse
import tifffile as tiff
from os.path import basename, join
import numpy as np
import os
import track_operation
from utils.file_io import make_dirs, imsave, lbread
from utils.parser import ParamParser
from utils.global_holder import holder
import logging

logger = logging.getLogger(__name__)


def neg2poslabels(labels):
    if not hasattr(holder, 'max'):
        holder.max = labels.max()
    holder.max = max(holder.max, labels.max())
    negatives = np.unique(labels[labels < 0])
    for i in negatives:
        holder.max += 1
        labels[labels == i] = holder.max
    return labels


def caller(inputs, inputs_labels, output, functions, params):
    make_dirs(output)
    img0, labels0 = imread(inputs[0]), lbread(inputs_labels[0]).astype(np.int16)
    labels0 = neg2poslabels(labels0)
    imsave(labels0, output, basename(inputs[0]), dtype=np.int16)
    for holder.frame, (path, pathl) in enumerate(zip(inputs[1:], inputs_labels[1:])):
        img1, labels1 = imread(path), lbread(pathl)
        labels1 = -labels1
        for fnum, (function, param) in enumerate(zip(functions, params)):
            func = getattr(track_operation, function)
            if not (labels1 < 0).any():
                continue
            labels0, labels1 = func(img0, img1, labels0, -labels1, **param)
            logger.debug('\t{0} with {1}: {2}'.format(function, param, len(set(labels1[labels1 < 0]))))
        logger.info("\tframe {0}: {1} objects linked and {2} unlinked.".format(holder.frame,
                    len(set(labels1[labels1 > 0])), len(set(labels1[labels1 < 0]))))
        labels0 = neg2poslabels(labels1)
        img0 = img1
        imsave(labels0, output, path, dtype=np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="images", nargs="+")
    parser.add_argument("-l", "--labels", help="labels", nargs="+")
    parser.add_argument("-o", "--output", help="output directory",
                        type=str, default='temp')
    parser.add_argument("-f", "--functions", help="functions", nargs="+")
    parser.add_argument('-p', '--param', nargs='+', help='parameters', action='append')
    args = parser.parse_args()

    if args.functions is None:
        print help(track_operation)
        return

    params = ParamParser(args.param).run()
    holder.args = args

    caller(args.input, args.labels, args.output, args.functions, params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
