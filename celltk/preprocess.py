"""
Any operations to make img from img.

python celltk/preprocess.py -f gaussian_laplace -i c0/img_00000000*
"""


# from scipy.ndimage import imread
import argparse
from utils.file_io import make_dirs, imsave
from utils.util import imread
from utils.parser import ParamParser, parse_image_files
import logging
from utils.global_holder import holder
import preprocess_operation

logger = logging.getLogger(__name__)


def caller(inputs, output, functions, params):
    holder.inputs = inputs
    make_dirs(output)

    logger.info("Functions {0} for {1} images.".format(functions, len(inputs)))

    for holder.frame, holder.path in enumerate(inputs):
        img = imread(holder.path)
        for function, param in zip(functions, params):
            func = getattr(preprocess_operation, function)
            img = func(img, **param)
        imsave(img, output, holder.path)
        logger.info("\tframe {0} done.".format(holder.frame))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="images", nargs="*")
    parser.add_argument("-o", "--output", help="output directory", type=str, default='temp')
    parser.add_argument("-f", "--functions", help="functions", nargs="*")
    parser.add_argument('-p', '--param', nargs='+', help='parameters', action='append')
    # parser.add_argument("-p", "--param", nargs="*", help="parameters", default=[])

    args = parser.parse_args()

    if args.functions is None:
        # print help(preprocess_operation)
        return

    params = ParamParser(args.param).run()
    args.input = parse_image_files(args.input)
    holder.args = args

    caller(args.input, args.output, args.functions, params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
