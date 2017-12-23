# -*- coding: utf-8 -*-
"""MatConvNet Importer module

The module contains methods to convert MatConvNet models (stored in either
the `simplenn` or `dagnn` formats) into PyTorch models. The model conversion
consists of two stages:
  (1) source code generation - a python file is generated containing the
      the network definition
  (2) state_dict construction - a dictionary of network weights that accompany
      the network defintion is stored to disk

-----------------------------------------------------------
Licensed under The MIT License [see LICENSE.md for details]
Copyright (C) 2017 Samuel Albanie
-----------------------------------------------------------
"""

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np

import ptmcn_utils as pmu
# clean-up:
# from utils import int_list, convert_padding, parse_struct

def build_header_str(net_name, debug_mode):
    header = '''
import torch
import torch.nn as nn

class {0}(nn.Module):

    def __init__(self):
        super().__init__()
'''
    if debug_mode:
        header = header + '''
        from collections import OrderedDict
        self.debug_feats = OrderedDict()
'''
    return header.format(net_name)

def build_forward_str(input_vars):
    forward_str = '''
    def forward(self, {}):
'''.format(input_vars)
    return forward_str

def build_forward_debug_str(input_vars):
    forward_debug_str = '''
    def forward_debug(self, {}):
        """ This purpose of this function is to provide an easy debugging
        utility for the converted network.  Cloning is used to prevent in-place
        operations from modifying feature artefacts. You can prevent the
        generation of this function by setting `debug_mode = False` in the
        importer tool.
    """
'''.format(input_vars)
    return forward_debug_str

def build_loader(net_name):
    loader_name = net_name.lower()
    forward_str = '''
def {0}(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = {1}()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model
'''.format(loader_name, net_name)
    return forward_str

def load_mcn_net(path):
    mcn = sio.loadmat(path)
    # sanity check
    for key in ['meta', 'params', 'layers']:
        assert key in mcn.keys()
    mcn_net = {
        'meta': pmu.parse_struct(mcn['meta']),
        'params': pmu.parse_struct(mcn['params']),
        'layers': pmu.parse_struct(mcn['layers']),
    }
    return mcn_net

def extract_dag(mcn_net):
    """ basic version assumes a linear chain """
    nodes = []
    num_layers = len(mcn_net['layers']['name'])
    for ii in range(num_layers):
        params = mcn_net['layers']['params'][ii]
        if params == {'': []}: params = None
        node = {
            'name': mcn_net['layers']['name'][ii],
            'inputs': mcn_net['layers']['inputs'][ii],
            'outputs': mcn_net['layers']['outputs'][ii],
            'params': params,
        }
        bt = mcn_net['layers']['type'][ii]
        block = mcn_net['layers']['block'][ii]
        opts = {'block': block, 'block_type': bt}
        if bt == 'dagnn.Conv':
            mod = pmu.conv2d_mod(block)
        elif bt == 'dagnn.ReLU':
            mod = nn.ReLU()
        elif bt == 'dagnn.Pooling':
            pad, ceil_mode = pmu.convert_padding(block['pad'])
            pool_opts = {'kernel_size': pmu.int_list(block['poolSize']),
                         'stride': pmu.int_list(block['stride']),
                         'padding': pad, 'ceil_mode': ceil_mode}
            if block['method'] == 'avg':
                mod = nn.AvgPool2d(**pool_opts)
            elif block['method'] == 'max':
                mod = nn.MaxPool2d(**pool_opts)
            else:
                msg = 'unknown pooling type: {}'.format(block['method'])
                raise ValueError(msg)
        elif bt == 'dagnn.DropOut': # both frameworks use p=prob(zeroed)
            mod = nn.Dropout(p=block['rate'])
        elif bt == 'dagnn.Permute':
            mod = pmu.Permute(**opts)
        elif bt == 'dagnn.Flatten':
            mod = pmu.Flatten(**opts)
        elif bt == 'dagnn.Concat':
            mod = pmu.Concat(**opts)
        node['mod'] = mod
        nodes += [node]
    return nodes

def cleanup(x):
    """fix unusual spacing present in nn.Module __repr__"""
    x = x.replace('Conv2d (', 'Conv2d(')
    return x

def ensure_compatible_repr(mod):
    """Fix minor bug in __repr__ function for MaxPool2d (present in older
    PyTorch versions)

    Args:
        mod (nn.Module): candidate module

    Returns:
        (str): modified, compatible __repr__ string for module
    """
    repr_str = str(mod)
    if isinstance(mod, nn.MaxPool2d):
        if not 'ceil_mode' in repr_str: # string manipulation to fix bug
            assert repr_str[-2:] == '))', 'unexpected repr format'
            repr_str = repr_str[:-1] + ', ceil_mode={})'.format(mod.ceil_mode)
    return repr_str

class Network(nn.Module):
    def __init__(self, name, mcn_net, debug_mode=True):
        super().__init__()
        self.name = name.capitalize()
        self.attr_str = []
        self.forward_str = []
        self.mcn_net = mcn_net
        self.input_vars = None
        self.output_vars = None
        self.forward_debug_str = []
        self.debug_mode = debug_mode

    def indenter(self, x, depth=2):
        num_spaces = 4
        indent = ' ' * depth * num_spaces
        return indent + '{}\n'.format(x)

    def forward_return(self):
        return 'return {}'.format(self.output_vars)

    def add_mod(self, name, inputs, outputs, params, mod, state_dict):
        if not isinstance(mod, pmu.PlaceHolder):
            mod_str = ensure_compatible_repr(mod)
            self.attr_str += ['self.{} = nn.{}'.format(name, mod_str)]
        outs = ','.join(outputs)
        ins = ','.join(inputs)
        if not self.input_vars: self.input_vars = ins
        self.output_vars = outs
        if isinstance(mod, pmu.PlaceHolder):
            func = str(mod).format(ins)
            forward_str = '{} = {}'.format(outs, func)
        else:
            forward_str = '{} = self.{}({})'.format(outs, name, ins)
        self.forward_str += [forward_str]
        if self.debug_mode:
            self.forward_debug_str += [forward_str]
            template = "self.debug_feats['{0}'] = torch.clone({0})"
            forward_debug_str = template.format(outs)
            self.forward_debug_str += [forward_debug_str]

        if not params: return # module has no associated params
        for idx, param_name in enumerate(params):
            if idx == 0:
                key = '{}.weight'.format(name)
            elif idx == 1:
                key = '{}.bias'.format(name)
            else:
                raise ValueError('unexpected number of params')
            val_idx = self.mcn_net['params']['name'].index(param_name)
            weights = self.mcn_net['params']['value'][val_idx]
            state_dict[key] = pmu.weights2tensor(weights)

    def transcribe(self, depth=2):
        """generate pytorch source code for the model"""
        assert self.input_vars, 'input vars must be set before transcribing'
        arch = build_header_str(self.name, self.debug_mode)
        for x in self.attr_str:
            arch += self.indenter(x, depth)
        arch += build_forward_str(self.input_vars)
        for x in self.forward_str:
            arch += self.indenter(x, depth)
        arch += self.indenter(self.forward_return(), depth)
        arch += build_forward_debug_str(self.input_vars)
        if self.debug_mode:
            for x in self.forward_debug_str:
                arch += self.indenter(x, depth)
        arch += build_loader(self.name)
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
        if isinstance(node['mod'], pmu.Flatten) \
            and isinstance(prev['mod'], pmu.Permute) \
            and np.array_equal(prev['mod'].order, [1, 0, 2, 3]): # perform merge
            new_node = {'name': node['name'], 'inputs': prev['inputs'],
                        'outputs': node['outputs'], 'mod': pmu.Flatten(),
                        'params': None}
            simplified.append(new_node)
            skip = True
        elif skip:
            skip = False
        else:
            simplified.append(prev)
    return simplified

def build_network(mcn_path, name, debug_mode):
    """ convert a list of dag nodes into an architecture description

    NOTE: we can ensure a valid execution order by exploiting the provided
    ordering of the stored network
    """
    mcn_net = load_mcn_net(mcn_path)
    nodes = extract_dag(mcn_net)
    nodes = simplify_dag(nodes)
    state_dict = OrderedDict()
    net = Network(name, mcn_net, debug_mode)
    for node in nodes:
        net.add_mod(**node, state_dict=state_dict)
    return net, state_dict

demo_dir = '/users/albanie/coding/libs/pt/pytorch-mcn'
demo = 'squeezenet1_0-a815701f.pth'
demo_path = pjoin(demo_dir, demo)
mcn_dir = '/users/albanie/data/models/matconvnet'
model_name = 'squeezenet1_0-pt-mcn.mat'
mcn_path = pjoin(mcn_dir, model_name)
print('loading mcn model...')
dest_name = 'squeezenet1_0'
arch_def_path = 'models/{}.py'.format(dest_name)
weights_path = 'weights/{}.pth'.format(dest_name)
net, state_dict = build_network(mcn_path, name=dest_name, debug_mode=True)

with open(arch_def_path, 'w') as f:
    f.write(str(net))
torch.save(state_dict, weights_path)
