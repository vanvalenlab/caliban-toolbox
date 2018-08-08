from caller import _retrieve_caller_based_on_function
from utils.parser import ParamParser, parse_image_files
import argparse
from utils.global_holder import holder
from utils.file_io import make_dirs, imsave
from utils.util import imread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputs", help="images", nargs="*")
    parser.add_argument("-l", "--inputs_labels", help="images", nargs="*", default=None)
    parser.add_argument("-o", "--output", help="output directory", type=str, default='temp')
    parser.add_argument("-f", "--functions", help="functions", nargs="*")
    parser.add_argument("-p", "--param", nargs="*", help="parameters", default=[])
    args = parser.parse_args()

    params = ParamParser(args.param).run()
    inputs = parse_image_files(args.inputs)
    if args.inputs_labels is not None:
        args.inputs_labels = parse_image_files(args.inputs_labels)
    caller = _retrieve_caller_based_on_function(args.functions[0])

    if len(args.functions) == 1 and args.functions[0] == 'apply':
        pass
        # ch_names = operation['ch_names'] if 'ch_names' in operation else images
        # obj_names = operation['obj_names'] if 'obj_names' in operation else labels
        # caller(zip(*inputs), zip(*args.inputs_labels), args.output, obj_names, ch_names)
    elif args.inputs_labels is None:
        caller(inputs, args.output, args.functions, params=params)
    else:
        caller(inputs, args.inputs_labels, args.output, args.functions, params=params)

if __name__ == "__main__":
    main()
