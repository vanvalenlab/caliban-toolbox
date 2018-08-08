import re
import ast
import __builtin__
import ast
import argparse


def split_params(inputs):
    if "/" not in inputs:
        return [inputs]
    store = []
    li = []
    while inputs:
        element = inputs.pop(0)
        if element == "/":
            store.append(li)
            li = []
        else:
            li.append(element)
    store.append(li)
    return store


class ParamParser(object):
    def __init__(self, param_args):
        self.param_args = param_args    

    def run(self):
        if self.param_args is None:
            return [{}]
        parameters = split_params(self.param_args[0])
        return [self.convert2dict(p) for p in parameters]

    def convert2dict(self, param):
        dictargs = dict(e.split('=') for e in param)
        for key, value in dictargs.iteritems():
            if value[0].isdigit():
                dictargs[key] = ast.literal_eval(value)
            elif value[0]=='[':
                try:
                    dictargs[key] = ast.literal_eval(value)
                except:
                    temp = []
                    for i in value[1:-1].split(','):
                        if i.isdigit():
                            temp.append(ast.literal_eval(i))
                        else:
                            temp.append(i)
                    dictargs[key] = temp
        return dictargs
        

class ParamParser1(object):
    def __init__(self, param_args):
        self.param_args = param_args

    def run(self):
        params = self.split_params(self.param_args[:])
        params = self.iter_combine_list(params)
        param_list = []
        for param in params:
            param_list.append(self.add_quotation(param))
        dict_list = [dict(i) for i in param_list]
        return self.iter_literal(dict_list)

    def iter_literal(self, dict_list):
        for dic in dict_list:
            for key, value in dic.iteritems():
                dic[key] = ast.literal_eval(value)
        return dict_list

    def iter_combine_list(self, params):
        store = []
        for param in params:
            store.append(self.combine_list(param))
        return store

    def add_quotation(self, ps):
        """
        """
        pss = []
        for p in ps:
            pair = p.split("=")
            if not pair[1] in dir(__builtin__):
                pair[1] = re.sub('([a-zA-Z/_][a-zA-Z0-9/._\*]*)', "'\g<1>'", pair[1])
            pss.append(pair)
        return pss

    def split_params(self, inputs):
        if "/" not in inputs:
            return [inputs]
        store = []
        li = []
        while inputs:
            element = inputs.pop(0)
            if element == "/":
                store.append(li)
                li = []
            else:
                li.append(element)
        store.append(li)
        return store

    def combine_list(self, param):
        if len(param) == 1:
            return param        
        itr = iter(param)
        ps = []
        while True:
            try:
                p = itr.next()
                if "[" in p:
                    temp = []
                    temp.append(p)
                    while True:
                        p = itr.next()
                        temp.append(p)
                        if "]" in p:
                            break
                    ps.append("".join(temp))
                else:
                    ps.append(p)
            except StopIteration:
                break
        return ps


def parse_image_files(inputs):
    if "/" not in inputs:
        return inputs
    store = []
    li = []
    while inputs:
        element = inputs.pop(0)
        if element == "/":
            store.append(li)
            li = []
        else:
            li.append(element)
    store.append(li)
    return zip(*store)