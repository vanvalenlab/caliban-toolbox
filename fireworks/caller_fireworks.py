'''
You need to add covertrack directory to PYTHONPATH
e.g. export PYTHONPATH="$PYTHONPATH:$PI_SCRATCH/kudo/covertrack"
'''

import os
import sys
from os.path import join, basename, exists, dirname
celltkroot = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(join(celltkroot, 'celltk'))
# sys.path.insert(0, join(celltkroot, 'celltk'))
from fireworks import FireTaskBase, explicit_serialize
from fireworks import Firework, LaunchPad, Workflow
from caller import parse_args, load_yaml, call_operations
import yaml
import collections


def convert(data):
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data


@explicit_serialize
class clustercelltk(FireTaskBase):
    _fw_name = "clustercelltk"
    required_params = ["contents"]

    def run_task(self, fw_spec):
        print "Running CellTK with input {0}".format(self["contents"])
        parallel_call(self["contents"])


def initiate_cluster(inputs):
    # check how many image folders are there
    contents_list = multi_call(inputs)
    lpad = LaunchPad(**yaml.load(open(join(celltkroot, "fireworks", "my_launchpad.yaml"))))
    wf_fws = []
    for contents in contents_list:
        fw_name = "cluster_celltk"
        fw = Firework(clustercelltk(contents=contents),
                      name = fw_name,
                      spec = {"_queueadapter": {"job_name": fw_name, "walltime": "47:00:00"}},
                      )
        wf_fws.append(fw)
    # end loop over input values
    workflow = Workflow(wf_fws, links_dict={})
    lpad.add_wf(workflow)


class parallel_call():
    def __init__(self, contents):
        contents = convert(contents)
        call_operations(contents)


def multi_call(inputs):
    contents = load_yaml(inputs)
    pin = contents['PARENT_INPUT']
    pin = pin[:-1] if pin.endswith('/') or pin.endswith('\\') else pin
    input_dirs = [join(pin, i) for i in os.listdir(pin)]
    contents_list = []
    for subfolder in input_dirs:
        conts = eval(str(contents).replace('$INPUT', subfolder))
        conts['OUTPUT_DIR'] = join(conts['OUTPUT_DIR'], basename(subfolder))
        contents_list.append(conts)
    return contents_list


def main():
    args = parse_args()
    initiate_cluster(args.input[0])


if __name__ == "__main__":
    main()
