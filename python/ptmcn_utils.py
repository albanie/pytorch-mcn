# -*- coding: utf-8 -*-
"""Utilities for MatConvNet model importer

-----------------------------------------------------------
Licensed under The MIT License [see LICENSE.md for details]
Copyright (C) 2017 Samuel Albanie
-----------------------------------------------------------
"""

import math
import numpy as np
import torch
import torch.nn as nn

def conv2d_mod(block):
    """Build a torch conv2d module from a matconvnet convolutional block

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

    Args:
        block (dict): a dictionary of mcn layer block attributes, parsed
        from the stored network .mat file.

    Returns:
        nn.Conv2d : the corresponding PyTorch convolutional module
    """
    fsize = int_list(block['size'])
    stride = tuple(int_list(block['stride']))
    assert len(fsize) == 4, 'expected four dimensions'
    pad, _ = convert_padding(block['pad'])
    conv_opts = {'in_channels': fsize[2], 'out_channels': fsize[3],
                 'kernel_size': fsize[:2], 'padding': pad, 'stride': stride}
    return nn.Conv2d(**conv_opts)

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
    # handle scalar base cases for recursion
    scalar_types = [np.str_, np.uint8, np.int64, np.float32, np.float64]
    if x.dtype.type in scalar_types:
        if x.size == 1: x = x.flatten() ; x = x[0] # flatten scalars
        return x
    fieldnames = [tup[0] for tup in x.dtype.descr]
    parsed = {f:[] for f in fieldnames}
    if any([dim == 0 for dim in x.shape]):
        return parsed #only recurse on valid storage shapes
    if fieldnames == ['']:
        return [elem[0] for elem in x.flatten()] # prevent blank nesting
    for f_idx, fname in enumerate(fieldnames):
        x = x.flatten() #implify logic via linear index
        if x.size > 1:
            for ii in range(x.size):
                parsed[fname].append(parse_struct(x[ii][f_idx]))
        else:
            parsed[fname] = parse_struct(x[0][f_idx])
    return parsed


def int_list(x):
    """Convert a (single-dimensional) numpy array of int-types to a native
     Python list of ints

    NOTE: As a general rule, pytorch constructors do not accepted numpy integer
    types as arguments.  It is therefore often easier to convert small
    numpy arrays (typically those corresponding to layer options) to native
    lists of Ints
    """
    if len(x.shape) > 1:
        assert sorted(x.shape)[-2] <= 1, 'invalid for multidim arrays'
    return x.flatten().astype(int).tolist()

def convert_padding(mcn_pad):
    """Transform matconvnet block padding to match pytorch padding conventions

    NOTE: mcn padding convention is [TOP BOTTOM LEFT RIGHT]. PyTorch module
    padding (when attached to a layer such as convolution, rather than the
    dedicated Padding nn.Module), contains two elements, in the format
    [PAD_Y, PAD_X], to be appended to both sides of the input. If the padding
    is not homogeous, e.g. the mcn padding `[0, 1, 0, 1]`, then `ceil_mode`
    can be used to adjust the output of the correspodning PyTorch module.

    Args:
        mcn_pad (ndaray): a 1x4 array specifying the matconvnet padding

    Returns:
        List, bool: a two-element list of integers defining the vertical and
        horizontal padding, together with a boolean value controlling whether
        to use ceil_mode.
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
    """Convert uniform mcn padding to pytorch pooling padding conventions

    Args:
        mcn_pad (ndaray): a 1x4 array specifying symmetric matconvnet padding

    Returns:
        List: a two-element list of integers defining the vertical and
        horizontal padding
    """
    assert len(mcn_pad) == 4
    assert mcn_pad[0] == mcn_pad[1], 'padding must be symmetric'
    assert mcn_pad[2] == mcn_pad[3], 'padding must be symmetric'
    if np.unique(mcn_pad).size == 1:
        pad = mcn_pad[0]
    else:
        pad = mcn_pad[:2]
    return pad

class PlaceHolder(object):
    """A placeholder class for pytorch operations that are defined through
    code execution, rather than as nn modules"""

    def __init__(self, block, block_type):
        self.block = block
        self.block_type = block_type

class Concat(PlaceHolder):
    """A class that represents the torch.cat() operation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mcn_order = [4, 3, 1, 2] #¬nsform cat dimension
        self.dim = mcn_order.index(int(self.block['dim']))

    def __repr__(self):
        return 'torch.cat(({{}}), dim={})'.format(self.dim)

class Flatten(PlaceHolder):
    """A class that represents the torch tesnor.view() operation"""

    def __init__(self, **kwargs):
        pass # void calling super

    def __repr__(self):
        return '{0}.view({0}.size(0), -1)'

class Permute(PlaceHolder):
    """A class that represents the torch.tranpose() operation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = self.block['order'].flatten() - 1 # fix 1-indexing

    def __repr__(self):
        #TODO(samuel): add logic for higher dims
        changes = self.order - np.arange(4)
        error_msg = 'only two dims can be transposed at a time'
        assert (changes == 0).sum() <= 2, error_msg
        error_msg = 'only tranpose along first dimensions currently supported'
        assert np.array_equal(np.where(changes != 0)[0], [0, 1]), error_msg
        dim0 = np.where(self.order == 0)[0][0]
        dim1 = np.where(self.order == 1)[0][0]
        return 'torch.tranpose({{}}, {}, {})'.format(dim0, dim1)

def weights2tensor(x):
    """Adjust memory layout and load weights as torch tensor

    Args:
        x (ndaray): a numpy array, corresponding to a set of network weights
        stored in column major order

    Returns:
        torch.tensor: a permuted sets of weights, matching the pytorch layout
        convention
    """
    if x.ndim == 4:
        x = x.transpose((3, 2, 0, 1))
    elif x.ndim == 2:
        if x.shape[1] == 1:
            x = x.flatten()
    return torch.from_numpy(x)

def build_header_str(net_name, debug_mode):
    """Generate source code header - constructs the header source
    code for the network definition file.

    Args:
        net_name (str): name of the network architecture
        debug_mode (bool): whether to generate additional debugging code

    Returns:
        (str) : source code header string.
    """
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
