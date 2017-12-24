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

import os
from collections import OrderedDict

import argparse
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np

import ptmcn_utils as pmu
import source_gen as sg

def load_mcn_net(path):
    """Load matconvnet network into Python dict

    By default, loading from a `.mat` file directly in Python produces
    a complicated ndarray structure.  This function constructs a simpler
    representation of the network for processing.

    Args:
        path (str): the path to the stored matconvnet network

    Returns:
        (dict): a native Python dictionary containg the network parameters,
        layers and meta information.
    """
    mcn = sio.loadmat(path)
    for key in ['meta', 'params', 'layers']: assert key in mcn.keys()
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

# def cleanup(repr_):
    # """fix unusual spacing present in nn.Module __repr__
    
    # Args:
        
    # """
    # return x

def ensure_compatible_repr(mod):
    """Fixes minor bug in __repr__ function for MaxPool2d (present in older
    PyTorch versions). This function also ensures more consistent repr str
    spacing.

    Args:
        mod (nn.Module): candidate module

    Returns:
        (str): modified, compatible __repr__ string for module
    """
    repr_str = str(mod)
    repr_str = repr_str.replace('Conv2d (', 'Conv2d(') # fix spacing issue
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
        arch = sg.build_header_str(self.name, self.debug_mode)
        for x in self.attr_str:
            arch += self.indenter(x, depth)
        arch += sg.build_forward_str(self.input_vars)
        for x in self.forward_str:
            arch += self.indenter(x, depth)
        arch += self.indenter(self.forward_return(), depth)
        if self.debug_mode:
            arch += sg.build_forward_debug_str(self.input_vars)
            for x in self.forward_debug_str:
                arch += self.indenter(x, depth)
        arch += sg.build_loader(self.name)
        # arch = cleanup(arch)
        return arch

    def __str__(self):
        return self.transcribe()

def simplify_dag(nodes):
    """Simplify unnecessary chains of operations

    Certain combinations of MCN operations can be simplified to a single
    PyTorch operation.  For example, because matlab tensors are stored in
    column major order, the common `x.view(x.size(0),-1)` function maps to
    a combination of `Permute` and `Flatten` layers.

    Args:
        nodes (List): A directed acyclic network, represented as a list
        of nodes, where each node is a dict containing layer attributes and
        pointers to its parents and children.

    Returns:
        (List): a list of nodes, in which a subset of the operations have
        been simplified.
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
    """Convert a list of dag nodes into an architecture description

    NOTE: We can ensure a valid execution order by exploiting the provided
    ordering of the stored network.

    Args:
        mcn_path (str): the path to the .mat file containing the matconvnet
          network to be converted.
        name (str): the name that will be given to the converted network.
        debug_mode (bool): If enabled, will generate additional source code
          that makes it easy to compute intermediate network features.

    Returns:
        (str, dict): A string comprising the generated source code for the
        network description, and a dictionary containing the associated
        netowrk parameters.
    """
    mcn_net = load_mcn_net(mcn_path)
    nodes = extract_dag(mcn_net)
    nodes = simplify_dag(nodes)
    state_dict = OrderedDict()
    net = Network(name, mcn_net, debug_mode)
    for node in nodes:
        net.add_mod(**node, state_dict=state_dict)
    return net, state_dict

def import_model(mcn_model_path, output_dir, refresh, debug_mode):
    """Import matconvnet model to pytorch

    Args:
        mcn_path (str): the path to the .mat file containing the matconvnet
          network to be converted.
        output_dir (str): path to the location where the imported pytorch model
          will be stored.
        refresh (bool): whether to overwrite existing imported models.
        debug_mode (bool): whether to generate additional model code to help
          with debugging.
    """
    print('Loading mcn model from {}...'.format(mcn_model_path))
    dest_name = os.path.splitext(os.path.basename(mcn_model_path))[0]
    dest_name = dest_name.replace('-', '_') # ensure valid python variable name
    arch_def_path = '{}/{}.py'.format(output_dir, dest_name)
    weights_path = '{}/{}.pth'.format(output_dir, dest_name)
    exists = all([os.path.exists(p) for p in [arch_def_path, weights_path]])
    if exists and not refresh:
        template = 'Found existing imported model at {},{}, skipping...'
        print(template.format(arch_def_path, weights_path))
        return
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    net, state_dict = build_network(mcn_model_path, dest_name, debug_mode)
    print('Saving imported model definition to {}'.format(arch_def_path))
    with open(arch_def_path, 'w') as f:
        f.write(str(net))
    print('Saving imported weights to {}'.format(weights_path))
    torch.save(state_dict, weights_path)

parser = argparse.ArgumentParser(
    description='Convert model from MatConvNet to PyTorch.')
parser.add_argument('mcn_model_path',
                    type=str,
                    help='The input should be the path to a matconvnet model \
                        file (a .mat) file, stored in either dagnn or simplenn \
                        format')
parser.add_argument('output_dir',
                    type=str,
                    default='./output',
                    help='Output MATLAB file')
parser.add_argument('--refresh', dest='refresh', action='store_true',
                    help='Overwrite existing imported models')
parser.add_argument('--debug_mode', dest='debug_mode', action='store_true',
                    help='Generate additional code to help with debugging')
parser.set_defaults(refresh=False)
parser.set_defaults(debug_mode=False)
parsed = parser.parse_args()

mcn_model_path = os.path.expanduser(parsed.mcn_model_path)
output_dir = os.path.expanduser(parsed.output_dir)
debug_mode = parsed.debug_mode
refresh = parsed.refresh

# run importer
import_model(mcn_model_path, output_dir, refresh, debug_mode)
