# -*- coding: utf-8 -*-
"""Import script

Pooling
---

"""

from os.path import join as pjoin
import torch.nn as nn
# import numpy as np
import functools
import scipy.io as sio
# build_header
import math
import numpy as np

# clean-up:
# from utils import int_list, convert_padding, parse_struct

def parse_struct(x):
    """Extract nested dict data structure from matfile struct layout

    When matlab data structures are loaded into python from .mat files
    via the `scipy.io.loadmat` utility, it can be slightly awkward to
    process the resulting numpy array. To maintain consistency with
    matlab structs, the loader uses a significant level of nesting and
    dimension insertion. This function aims to simplify this data structure
    into a native python dict, minimising the nesting depth where possible
    and flattening excess dimensions.

    Args:
        x (ndarray): a nested, mixed type numpy array that has been
                     loaded from a .mat file with the scipy.io.loadmat
                     utility.
    Returns:
        nested dictionary of parsed data
    """
    # handle scalar base cases for recursion
    # if x.shape == (1,2): import ipdb ; ipdb.set_trace()
    scalar_types = [np.str_, np.uint8, np.int64, np.float32, np.float64]
    if x.dtype.type in scalar_types:
        if x.size == 1: x = x.flatten() ; x = x[0] # flatten scalars
        return x
    fieldnames = [tup[0] for tup in x.dtype.descr]
    parsed = {f:[] for f in fieldnames}
    if any([dim == 0 for dim in x.shape]):
        return parsed # only recurse on valid storage shapes
    if fieldnames == ['']:
        return [elem[0] for elem in x.flatten()] # prevent blank nesting
        # return x[0][0]
    # if fieldnames == [('', '|O')]: import ipdb ; ipdb.set_trace()
    # if 'inputs' in fieldnames: import ipdb ; ipdb.set_trace()
    for f_idx, fname in enumerate(fieldnames):
        x = x.flatten() # simplify logic via linear index
        if x.size > 1:
            for ii in range(x.size):
                parsed[fname].append(parse_struct(x[ii][f_idx]))
        else:
            parsed[fname] = parse_struct(x[0][f_idx])
    return parsed

def int_list(x):
    """As a general rule, pytorch constructors do not accepted numpy integer
    types as arguments.  It is therefore often easier to convert small
    numpy arrays (typically those corresponding to layer options) to native
    lists of Ints
    """
    if len(x.shape) > 1:
        assert sorted(x.shape)[-2] <= 1, 'invalid for multidim arrays'
    return x.flatten().astype(int).tolist()

def convert_padding(mcn_pad):
    """convert padding to pytorch padding conventions
    NOTE: mcn padding convention is [TOP BOTTOM LEFT RIGHT]
    """
    mcn_pad = int_list(mcn_pad)
    if mcn_pad[0] == mcn_pad[1] and mcn_pad[2] == mcn_pad[3]:
        pad = convert_uniform_padding(mcn_pad)
        ceil_mode = False
    else:
        assert math.fabs(mcn_pad[0] - mcn_pad[1]) <= 1, 'cannot be resolved'
        assert math.fabs(mcn_pad[2] - mcn_pad[3]) <= 1, 'cannot be resolved'
        pad = (min(mcn_pad[:2]), min(mcn_pad[2:]))
        ceil_mode = True
    return pad, ceil_mode

def convert_uniform_padding(mcn_pad):
    """convert uniform mcn padding to pytorch pooling padding conventions
    """
    assert len(mcn_pad) == 4
    assert mcn_pad[0] == mcn_pad[1], 'padding must be symmetric'
    assert mcn_pad[2] == mcn_pad[3], 'padding must be symmetric'
    if np.unique(mcn_pad).size == 1:
        pad = mcn_pad[0]
    else:
        pad = mcn_pad[:2]
    return pad


def build_header_str(net_name):
    header = '''
import torch
import torch.nn as nn

class {0}(nn.module):

    def __init__(self):
        super().__init__()
'''
    return header.format(net_name)

def build_forward_str(input_vars):
    forward_str = '''
    def forward(self, {}):
'''.format(input_vars)
    return forward_str

# class TorchModule(object):
    # def __init__(self, name, inputs, outputs):
	# self.name = name
	# self.inputs = inputs
	# self.outputs = outputs
	# self.params = []
	# self.model = None

# def build_conv2d(block,

# class Conv2d(TorchModule):
    # def __init__(self, name, inputs, outputs):
	# self.name = name
	# self.inputs = inputs
	# self.outputs = outputs
	# self.params = []
	# self.model = None

def conv2d_mod(block):
    """
    build a torch conv2d module from a matconvnet convolutional block

    NOTES:
    PyTorch and Matconvnet use similar padding conventions for convolution.
    For the description below, assume that a convolutional block has input `X`,
    filters `f` biases `b` and output `Y` with dimensions as follows:
        X: H x W x D
        f: H' x W' x D x D''
        b: D'' x 1
        Y: H'' x W'' x D''
    Both Matconvnet and PyTorch compute "valid" convolutions - they only
    produce outputs at locations which possess complete filter support in the
    input.

    Assuming the filters have dilation [DIL_Y DIL_X], then the output size for
    both frameworks is given by
    H'' = 1 + floor((H - DIL_Y * (H'-1) + P_H)/S[0])
    W'' = 1 + floor((W - DIL_X * (W'-1) + P_W)/S[1])

    In PT, the padding P is a tuple of two numbers, so that P_H = 2 * P[0] and
    P_W = 2 * P[1].  In MCN, the padding can be an array of four numbers, so
    that P_H = P[0] + P[1] and  P_W = P[2] + P[3] (here P has the format
    [TOP BOTTOM LEFT RIGHT]).
    """
    fsize = int_list(block['size'])
    assert len(fsize) == 4, 'expected four dimensions'
    pad,_ = convert_padding(block['pad'])
    conv_opts = {'in_channels': fsize[2], 'out_channels': fsize[3],
                 'kernel_size': fsize[:2], 'padding': pad}
    return nn.Conv2d(**conv_opts)

def load_mcn_net(path):
    mcn = sio.loadmat(path)
    # sanity check
    for key in ['meta', 'params', 'layers']:
        assert key in mcn.keys()
    mcn_net = {
        'meta': parse_struct(mcn['meta']),
        'params': parse_struct(mcn['params']),
        'layers': parse_struct(mcn['layers']),
    }
    return mcn_net

class PlaceHolder(object):
    """placeholder class for pytorch operations that are defined through
    code execution, rather than as nn modules"""

    def __init__(self, block, block_type):
        self.block = block
        self.block_type = block_type

class Concat(PlaceHolder):

    def __repr__(self):
        return 'torch.cat({{}}, dim={})'.format(int(self.block['dim']))

class Flatten(PlaceHolder):

    def __init__(self, **kwargs):
        pass

    def __repr__(self):
        return 'torch.view({{}}.size(0), -1)'

class Permute(PlaceHolder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = self.block['order'].flatten() - 1 # fix 1-indexing

    def __repr__(self):
        #TODO(samuel): add logic for higher dims
        changes = self.order - np.arange(4)
        error_msg = 'only two dims can be transposed at a time'
        assert (changes == 0).sum() <= 2, error_msg
        error_msg = 'only tranpose along first dimensions currently supported'
        assert np.array_equal(np.where(changes != 0)[0], [0,1]), error_msg
        dim0 = np.where(self.order == 0)[0][0]
        dim1 = np.where(self.order == 1)[0][0]
        return 'torch.tranpose({{}}, {}, {})'.format(dim0, dim1)

class View(PlaceHolder):
    """TODO: fix general case"""

    def __init__(self, flatten=True):
        self.flatten = flatten

    def __repr__(self):
        if not self.flatten: raise ValueError('not yet supported')
        return '{}.view({}.size(0), -1)'

# ['dagnn.Flatten',
         # 'dagnn.Conv',
          # 'dagnn.Concat',
           # 'dagnn.Permute',
            # 'dagnn.ReLU',
             # 'dagnn.DropOut',
              # 'dagnn.Pooling']

def extract_dag(mcn_net):
    """ basic version assumes a linear chain """
    nodes = []
    num_layers = len(mcn_net['layers']['name'])
    for ii in range(num_layers):
        node = {
           'name': mcn_net['layers']['name'][ii],
           'inputs': mcn_net['layers']['inputs'][ii],
           'outputs': mcn_net['layers']['outputs'][ii],
        }
        bt = mcn_net['layers']['type'][ii]
        block = mcn_net['layers']['block'][ii]
        opts = {'block': block, 'block_type': bt}
        if bt == 'dagnn.Conv':
            mod = conv2d_mod(block)
        elif bt == 'dagnn.ReLU':
            mod = nn.ReLU()
        elif bt == 'dagnn.Pooling':
            pad, ceil_mode = convert_padding(block['pad'])
            pool_opts = {'kernel_size': int_list(block['poolSize']),
                         'stride': int_list(block['stride']),
                         'padding': pad, 'ceil_mode': ceil_mode}
            if block['method'] == 'avg':
                mod = nn.MaxPool2d(**pool_opts)
            elif block['method'] == 'max':
                mod = nn.MaxPool2d(**pool_opts)
            else:
                msg = 'unknown pooling type: {}'.format(block['method'])
                raise ValueError(msg)
        elif bt == 'dagnn.DropOut': # both frameworks use p=prob(zeroed)
            mod = nn.Dropout(p=block['rate'])
        elif bt == 'dagnn.Permute':
            mod = Permute(**opts)
        elif bt == 'dagnn.Flatten':
            mod = Flatten(**opts)
        elif bt == 'dagnn.Concat':
            mod = Concat(**opts)
        # elif bt in ['dagnn.Permute', 'dagnn.Flatten', 'dagnn.Concat']:
            # import ipdb ; ipdb.set_trace()
            # mod = PlaceHolder(**opts)
        node['mod'] = mod
        nodes += [node]
    return nodes

def compose(*funcs):
    """compose a sequnce of functions into a single function

    Args :
        *funcs [iterable]: a sequence of functions
    Return:
        single, composite function
    """
    identity = lambda x: x
    compose = lambda f, g: lambda x: f(g(x))
    return functools.reduce(compose, funcs, identity)

def cleanup(x):
    """fix unusual spacing present in nn.module __repr__"""
    x = x.replace('Conv2d (', 'Conv2d(')
    return x

class Network(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.attr_str = []
        self.forward_str = []
        self.input_vars = None
        self.barbarian = False

    def indenter(self, x, depth=2):
        num_spaces = 4
        if self.barbarian: num_spaces = 2
        indent = ' ' * depth * num_spaces
        return indent + '{}\n'.format(x)

    def add_mod(self, name, inputs, outputs, mod):
        if not isinstance(mod, PlaceHolder):
            self.attr_str += ['self.{} = nn.{}'.format(name, mod)]
        outs = ','.join(outputs)
        ins = ','.join(inputs)
        if not self.input_vars: self.input_vars = ins
        if isinstance(mod, PlaceHolder):
            func = str(mod).format(ins)
            forward_str = '{} = {}'.format(outs, func)
        else:
            forward_str = '{} = self.{}({})'.format(outs, name, ins)
        self.forward_str += [forward_str]

    def transcribe(self):
        """generate pytorch source code for the model"""
        assert self.input_vars, 'input vars must be set before transcribing'
        arch = build_header_str(self.name)
        for x in self.attr_str:
            arch += self.indenter(x, depth=2)
        arch += build_forward_str(self.input_vars)
        for x in self.forward_str:
            arch += self.indenter(x, depth=2)
        arch = cleanup(arch)
        return arch

    def __str__(self):
        return self.transcribe()

def simplify_dag(nodes):
    """simplify unnecessary operations

    Certain combinations of MCN operations can be simplified to a single
    PyTorch operation.  For example, because matlab tensors are stored in
    column major order, the common `x.view(x.size(0),-1)` function maps to
    a combination of `Permute` and `Flatten` layers.

    """
    #TODO(samuel): clean up code
    simplified = []
    skip = False
    for prev, node in zip(nodes, nodes[1:]):
        if isinstance(node['mod'], Flatten) \
            and isinstance(prev['mod'], Permute) \
            and np.array_equal(prev['mod'].order, [1,0,2,3]): # perform merge
                new_node = {'name': node['name'], 'inputs': prev['inputs'],
                            'outputs': node['outputs'], 'mod': Flatten()}
                simplified.append(new_node)
                skip = True
        elif skip:
            skip = False
        else:
            simplified.append(prev)
    return simplified

def build_network(mcn_path, name):
    """ convert a list of dag nodes into an architecture description

    NOTE: we can ensure a valid execution order by exploiting the provided
    ordering of the stored network
    """
    mcn_net = load_mcn_net(mcn_path)
    nodes = extract_dag(mcn_net)
    nodes = simplify_dag(nodes)
    net = Network(name=name)
    for node in nodes:
        net.add_mod(**node)
    return net

# def mcn_to_pytorch(mcn_path,outputname=None):
    # mcn_net = load_mcn_net(mcn_path)
    # # if type(model).__name__=='hashable_uniq_dict': mcn_net=model.model
    # # model.gradInput = None
    # slist = extract_mcn_modules(mcn_net)
    # # s = simplify_source(slist)
    # header = build_header()

    # varname = t7_filename.replace('.t7','').replace('.','_').replace('-','_')
    # s = '{}\n\n{} = {}'.format(header,varname,s[:-2])

    # if not outputname: outputname = varname
    # arch_file = '{}.py'.format(out_name)
    # param_file = '{}.pth'.format(out_name)
    # with open(arch_file, "w") as f:
        # f.write(s)

    # net = nn.Sequential()
    # mcn_build_net(model,net)
    # torch.save(net.state_dict(), outputname + '.pth')

# def generate_arch():
# with open(arch_def_path, 'w') as f:
    # f.write(arch_def)

arch_def_path = 'squeezenet1_0-pt.py'
mcn_dir = '/users/albanie/data/models/matconvnet'
model_name = 'squeezenet1_0-pt-mcn.mat'
mcn_path = pjoin(mcn_dir, model_name)
print('loading mcn model...')
net = build_network(mcn_path, name='squeezenet1_0')

with open(arch_def_path, 'w') as f:
    f.write(str(net))
# mcn = sio.loadmat(mcn_path)
# net = Network(name='squeezenet1_0')
# for node in nodes: net.add_mod(**node)
# net = build_network(net, nodes)
# print(net)
