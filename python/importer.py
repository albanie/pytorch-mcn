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

def flatten_if_needed(nodes, complete_dag, is_flattened, flatten_layer):
    """Ensure that the DAG outputs a two dimensional tensor

    To maintain a consistent prediction interface, all converted pytorch
    networks output a 2d tensor (rather than a 4d tensor).  This function
    ensures that the DAG represented by the list of nodes meets this
    requirement.

    Args:
        nodes (List): A directed acyclic network, represented as a list
          of nodes, where each node is a dict containing layer attributes and
          pointers to its parents and children.
        complete_dag (bool): whehter of not the current list of nodes represents
          the complete DAG.
        is_flattened (bool): whether or not the DAG has already been "flattened"
          to produce tenors of the require dimensionality.
        flatten_layer (str): can be either the name of the layer after which
          flattening is performed, or 'last', in which case it is performed at
          the end of the network.

    Returns:
        (List): the updated DAG.
    """
    if not is_flattened:
        prev = nodes[-1]
        flatten_condition = (flatten_layer == 'last' and complete_dag) \
                or (flatten_layer == prev['name'])
        if flatten_condition:
            # import ipdb ; ipdb.set_trace()
            # TODO(samuel): Make network surgery more robust (it currently
            # relies on a unique variable naming modification to maintain a
            # consistent execution order)
            name = '{}_flatten'.format(prev['name'])
            outputs = prev['outputs']
            prev['outputs'] = ['{}_preflatten'.format(x)
                                        for x in prev['outputs']]
            node = {
                'name': name,
                'inputs': prev['outputs'],
                'outputs': outputs,
                'params': [],
            }
            node['mod'] = pmu.Flatten()
            nodes.append(node)
            is_flattened = True

        # if flatten_layer == 'last' and complete_dag:
            # name = '{}_flatten'.format(prev['name'])
            # outputs = [name]
            # node_pos = len(nodes)
        # elif flatten_layer == prev['name']:
            # import ipdb ; ipdb.set_trace()
            # node_names = [node['name'] for node in nodes]
            # idx = node_names.index(flatten_layer)
            # prev, follow = nodes[idx], nodes[idx + 1]
            # outputs = follow['inputs']
            # node_pos = idx + 1
        # else:
    return nodes, is_flattened

def extract_dag(mcn_net, drop_prob_softmax=True, in_ch=3, flatten_layer='last'):
    """Extract DAG nodes from stored matconvnet network

    Transform a stored mcn dagnn network structure into a Directed Acyclic
    Graph, represented as a list of nodes, each of which has pointers to
    its inputs and outputs. Since MatConvNet computes convolution groups
    online, rather than as a stored attribute (as done in PyTorch), the
    number of channels is tracked during DAG construction.

    Args:
        mcn_net (dict): a native Python dictionary containg the network
          parameters, layers and meta information.
        drop_prob_softmax (bool) [True]: whether to remove the final softmax
          layer of a network, if present.
        in_ch (int) [3]: the number of channels expected in input data
          processed by the network.
        flatten (str) [last]: the layer after which a "flatten" operation
          should be inserted (if one is not present in the matconvnet network).
    """
    # TODO(samuel): improve state management
    nodes = []
    is_flattened = False
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
            mod, in_ch = pmu.conv2d_mod(block, in_ch, is_flattened)
        elif bt == 'dagnn.BatchNorm':
            mod = pmu.batchnorm2d_mod(block, mcn_net, params)
        elif bt == 'dagnn.ReLU':
            mod = nn.ReLU()
        elif bt == 'dagnn.Pooling':
            pad, ceil_mode = pmu.convert_padding(block['pad'])
            pool_opts = {'kernel_size': pmu.int_list(block['poolSize']),
                         'stride': pmu.int_list(block['stride']),
                         'padding': pad, 'ceil_mode': ceil_mode}
            if block['method'] == 'avg':
                # mcn never includes padding in average pooling.
                # TODO(samuel): cleanup and add explanation
                pool_opts['count_include_pad'] = False
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
            is_flattened = True
        elif bt == 'dagnn.Concat':
            mod = pmu.Concat(**opts)
        elif bt == 'dagnn.Sum':
            mod = pmu.Sum(**opts)
        elif bt == 'dagnn.SoftMax' \
            and (ii == num_layers -1) and drop_prob_softmax:
            continue # remove softmax prediction layer
        else: import ipdb ; ipdb.set_trace()
        node['mod'] = mod
        nodes += [node]
        complete_dag = (ii == num_layers -1)
        nodes, is_flattened = flatten_if_needed(nodes, complete_dag,
                                                is_flattened, flatten_layer)
    return nodes

def ensure_compatible_repr(mod):
    """Fixes minor bug in __repr__ functions for certain modules (present in
	older PyTorch versions). This function also ensures more consistent repr
	str spacing.

    Args: mod (nn.Module): candidate module

    Returns:
        (str): modified, compatible __repr__ string for module
    """
    repr_str = str(mod)
    repr_str = repr_str.replace('Conv2d (', 'Conv2d(') # fix spacing issue
    if isinstance(mod, nn.MaxPool2d):
        if not 'ceil_mode' in repr_str: # string manipulation to fix bug
            assert repr_str[-2:] == '))', 'unexpected repr format'
            repr_str = repr_str[:-1] + ', ceil_mode={})'.format(mod.ceil_mode)
    elif isinstance(mod, nn.Linear):
        if not 'bias' in repr_str: # string manipulation to fix bug
            assert repr_str[-1:] == ')', 'unexpected repr format'
            bias = mod.bias is not None
            repr_str = repr_str[:-1] + ', bias={})'.format(bias)
    return repr_str

class Network(nn.Module):
    def __init__(self, name, mcn_net, meta, debug_mode=True):
        super().__init__()
        self.name = name.capitalize()
        self.attr_str = []
        self.meta = meta
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
            template = "self.debug_feats['{0}'] = {0}.clone()"
            forward_debug_str = template.format(outs)
            self.forward_debug_str += [forward_debug_str]
        if not params: return # module has no associated params
        for idx, param_name in enumerate(params):
            if idx == 0:
                key = '{}.weight'.format(name)
            elif idx == 1:
                key = '{}.bias'.format(name)
            elif idx == 2:
                msg = 'The third parameter should correspond to bn moments'
                assert 'moments' in params[idx], msg
                state_dict = pmu.update_bnorm_moments(name, param_name,
                                                     self.mcn_net, mod.eps,
                                                     state_dict)
                continue
            else:
                raise ValueError('unexpected number of params')
            val_idx = self.mcn_net['params']['name'].index(param_name)
            weights = self.mcn_net['params']['value'][val_idx]
            state_dict[key] = pmu.weights2tensor(weights)

    def transcribe(self, depth=2):
        """generate pytorch source code for the model"""
        assert self.input_vars, 'input vars must be set before transcribing'
        arch = sg.build_header_str(self.name, **self.meta,
                                   debug_mode=self.debug_mode)
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
            print('simplifying {}, {}'.format(prev['name'], node['name']))
        elif skip:
            skip = False
        else:
            simplified.append(prev)
    if not skip: simplified.append(node) # handle final node
    return simplified

def build_network(mcn_path, name, flatten_layer, debug_mode):
    """Convert a list of dag nodes into an architecture description

    NOTE: We can ensure a valid execution order by exploiting the provided
    ordering of the stored network.

    Args:
        mcn_path (str): the path to the .mat file containing the matconvnet
          network to be converted.
        name (str): the name that will be given to the converted network.
        flatten_layer (str): can be either the name of the layer after which
          flattening is performed, or 'last', in which case it is performed at
          the end of the network.
        debug_mode (bool): If enabled, will generate additional source code
          that makes it easy to compute intermediate network features.

    Returns:
        (str, dict): A string comprising the generated source code for the
        network description, and a dictionary containing the associated
        netowrk parameters.
    """
    mcn_net = load_mcn_net(mcn_path)
    normalization = mcn_net['meta']['normalization']
    if 'imageStd' in normalization:
        rgb_std = normalization['imageStd'].flatten().tolist()
    else:
        rgb_std = [1, 1, 1]
    meta = {'rgb_mean': normalization['averageImage'].flatten().tolist(),
            'rgb_std': rgb_std,
            'im_size': pmu.int_list(normalization['imageSize'])}
    nodes = extract_dag(mcn_net, flatten_layer=flatten_layer)
    nodes = simplify_dag(nodes)
    state_dict = OrderedDict()
    net = Network(name, mcn_net, meta, debug_mode=debug_mode)
    for node in nodes:
        net.add_mod(**node, state_dict=state_dict)
    return net, state_dict

def import_model(mcn_model_path, output_dir, refresh, **kwargs):
    """Import matconvnet model to pytorch

    Args:
        mcn_path (str): the path to the .mat file containing the matconvnet
          network to be converted.
        output_dir (str): path to the location where the imported pytorch model
          will be stored.
        refresh (bool): whether to overwrite existing imported models.

    Keyword Args:
        debug_mode (bool): whether to generate additional model code to help
          with debugging.
        flatten_layer (str): can be either the name of the layer after which
          flattening is performed, or 'last', in which case it is performed at
          the end of the network.
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
    net, state_dict = build_network(mcn_model_path, dest_name, **kwargs)
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
parser.add_argument('--flatten_layer', type=str, dest='flatten_layer',
                    help=('Supply name of MatConvNet layer after which a',
                          'PyTorch "(i.e. View(x, -1) for input x), operation',
                          'should be performed'))
parser.add_argument('--debug_mode', dest='debug_mode', action='store_true',
                    help='Generate additional code to help with debugging')
parser.set_defaults(refresh=False)
parser.set_defaults(debug_mode=False)
parser.set_defaults(flatten_layer='last')
parsed = parser.parse_args()

mcn_model_path = os.path.expanduser(parsed.mcn_model_path)
output_dir = os.path.expanduser(parsed.output_dir)
opts = {'flatten_layer': parsed.flatten_layer,
        'debug_mode': parsed.debug_mode,
        'refresh': parsed.refresh}

# run importer
import_model(mcn_model_path, output_dir, **opts)
