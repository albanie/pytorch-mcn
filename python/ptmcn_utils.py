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

def conv2d_mod(block, in_ch, is_flattened, verbose):
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

    Note that if both spatial dimensions of the kernel are 1, PyTorch has the
    option to use a linear module to perform a straight matrix multiply.

    Args:
        block (dict): a dictionary of mcn layer block attributes, parsed
        from the stored network .mat file.
        in_ch (int): the number of input channels to be processed by the
          convolution
        is_flattened (bool): if true, replace 1x1 convolutions with linear
          operators
        verbose (bool): whether to display more detailed information during the
          conversion process.

    Returns:

        (nn.Conv2d/nn.Linear, int) : the corresponding PyTorch convolutional
        module and the number of output channels produced by this module
    """
    fsize = int_list(block['size'])
    stride = tuple(int_list(block['stride']))
    assert len(fsize) == 4, 'expected four dimensions'
    pad, _ = convert_padding(block['pad'])
    if fsize[:2] == [1,1] and is_flattened:
        assert pad == 0, 'padding must be zero for linear layers'
        mod = nn.Linear(fsize[2], fsize[3], bias=bool(block['hasBias']))
    else:
        msg = 'valid convolution groups cannot be formed from filters'
        valid_conv_groups = int(in_ch / fsize[2]) == (in_ch / fsize[2])
        if not valid_conv_groups: import ipdb ; ipdb.set_trace()
        assert valid_conv_groups, msg
        num_groups = int(in_ch / fsize[2])
        conv_opts = {'in_channels': in_ch, 'out_channels': fsize[3],
                     'bias': block['hasBias'], 'kernel_size': fsize[:2],
                     'padding': pad, 'stride': stride, 'groups': num_groups}
        if not block['hasBias'] and verbose: print('skipping conv bias term')
        mod = nn.Conv2d(**conv_opts)
    return mod, fsize[3]

def align_interfaces(mcn_net, remap_aff_gen=True):
    """Modify layer inputs/outputs to match pytorch conventions

    Unlike in mcn, the interface for affine grid generation in pytorch
    requires knowledge of the batch size and number of channels of the
    feature map to which it will be applied. To allow the correct source
    code generation, this feature map is added as an additional input to
    the mcn graph through a simple graph search.  Since this assumes
    that the grid is going to be coupled to a Bilinear Sampler(which may not
    be the case in more unusual architectures, which may predict a grid without
    its application, it can be disabled).

        mcn_net (dict): a native Python dictionary containg the network
          parameters, layers and meta information.
        remap_aff_gen (bool, optional [True]): remap the inputs to the grid
          generator block to enable runtime estimation of grid size.

    Returns:
        (dict): the updated network
    """
    if not remap_aff_gen:
        return mcn_net
    layer_types = mcn_net['layers']['type']
    if not 'dagnn.AffineGridGenerator' in layer_types:
        return mcn_net
    # find the relevant layers
    aff_idx = layer_types.index('dagnn.AffineGridGenerator')
    outputs = mcn_net['layers']['outputs'][aff_idx]
    assert len(outputs) == 1, 'expected grid generator to have a single output'
    for c_idx, input_vars in enumerate(mcn_net['layers']['inputs']):
        if outputs[0] in input_vars:
            sampler_idx = c_idx # we assume the grid is consumed by a sampler
    sampler_inputs = mcn_net['layers']['inputs'][sampler_idx]
    assert len(sampler_inputs) ==  2, 'expected sampler to have two inputs'
    feat_name = sampler_inputs[0]
    # supply feature as additional input to affine grid generator to resolve
    # runtime size estimation issue
    mcn_net['layers']['inputs'][aff_idx].append(feat_name)
    return mcn_net

def batchnorm2d_mod(block, mcn_net, param_names):
    """Build a torch BatcNnorm2d module from a matconvnet BatchNorm block

    PyTorch batch normalization blocks require knowledge of the number of
    channels that the block will process - these are obtained by inspecting
    the parameters of the corresponding mcn block.  Unlike in mcn, pytorch
    stores the moments of the module as separate state, rather than parameters
    which requires that they are registered as buffers.

    Args:
        block (dict) : attributes for the mcn batch norm layer
        mcn_net (dict) : the full mcn network, stored in memory
        param_names (List[str]): the names of the parameters associated with
          the mcn batch norm layer.

    Returns:
        nn.BatchNorm2d : the corresponding PyTorch batch norm module
    """
    moment_names = param_names[2]
    val_idx = mcn_net['params']['name'].index(moment_names)
    moments = mcn_net['params']['value'][val_idx]
    num_channels = moments.shape[0]
    mod = nn.BatchNorm2d(num_channels, eps=block['epsilon'])
    return mod

def update_bnorm_moments(name, param_name, mcn_net, eps, state_dict):
    """Convert matconvnet batch norm moments into PyTorch equivalents

    PyTorch stores running variances, rather than running standard
    deviations (as done by mcn) when tracking batch norm moments, so
    these are converted accordingly. Both moments (means and stds)
    are stored as columns in a single mcn parameter array.

    PyTorch uses a slightly different formula for batch norm than
	mcn at test time (which incorporates the `eps` values during training
    rather than during inference):
	pytorch:
	   y = ( x - mean(x)) / (sqrt(var(x)) + eps) * gamma + beta
	mcn:
	   y = ( x - mean(x)) / sqrt(var(x)) * gamma + beta

    Args:
        name (str): the name of the bnorm layer to be processed
        param_name (str): the name of the mcn parameter containing the moments
        mcn_net (dict): the full mcn network, stored in memory
        eps (float): eps value (used in formula above)
        state_dict (dict): the dictionary of all states associated with the
          model

    Returns:
        dict : the updated state dictionary
    """
    val_idx = mcn_net['params']['name'].index(param_name)
    moments = mcn_net['params']['value'][val_idx]
    key = '{}.running_mean'.format(name)
    running_mean = moments[:, 0]
    state_dict[key] = weights2tensor(running_mean)
    running_vars = (moments[:,1] ** 2) - eps
    key = '{}.running_var'.format(name)
    state_dict[key] = weights2tensor(running_vars)
    return state_dict

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
    scalar_types = [np.str_, np.uint8, np.uint16, np.int64, np.float32, np.float64]
    if x.dtype.type in scalar_types:
        if x.size == 1: x = x.flatten() ; x = x[0] # flatten scalars
        return x
    fieldnames = [tup[0] for tup in x.dtype.descr]
    parsed = {f:[] for f in fieldnames}
    if any([dim == 0 for dim in x.shape]) or \
        (x.size == 1 and x.flatten()[0] is None):
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
        if math.fabs(mcn_pad[0] - mcn_pad[1]) > 1: import ipdb ; ipdb.set_trace()
        assert math.fabs(mcn_pad[0] - mcn_pad[1]) <= 1, 'cannot be resolved'
        assert math.fabs(mcn_pad[2] - mcn_pad[3]) <= 1, 'cannot be resolved'
        pad = (min(mcn_pad[:2]), min(mcn_pad[2:]))
        ceil_mode = True
    return pad, ceil_mode

def convert_uniform_padding(mcn_pad):
    """Convert uniform mcn padding to pytorch pooling padding conventions

    Here "uniform" refers to the fact that the same padding is applied to the
    top and the bottom of the input.  Similarly the same padding is applied to
    both the left and the right of the input.  However, the vertical and
    horizontal padding need not be identical.

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
        pad = tuple(mcn_pad[1:3])
    return pad

def globalpool_mod(block):
    """Build a torch pool module from a matconvnet GlobalPooling block

    Args:
        block (dict) : attributes for the mcn batch norm layer

    Returns:
        nn.Module : the corresponding global pooling module
    """
    method = block['method'].lower()
    if method == 'avg':
        mod = nn.AdaptiveAvgPool2d(1)
    elif method == 'max':
        mod = nn.AdaptiveMaxPool2d(1)
    else:
        raise ValueError('unrecognised pooling method {}'.format(method))
    return mod

class PlaceHolder(object):
    """A placeholder class for pytorch operations that are defined through
    code execution, rather than as nn modules

    Args:
        blank (bool) [False]: if True, the placeholder will produce no output
        in the final network architecture.
    Kwargs:
        All kwargs are set as attributes on the object.
    """

    def __init__(self, blank=False, **kwargs):
        self.blank = blank
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        raise ValueError('must be overriden by subclass')

class Concat(PlaceHolder):
    """A class that represents the torch.cat() operation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mcn_order = [4, 3, 1, 2] #¬nsform cat dimension
        self.dim = mcn_order.index(int(self.block['dim']))

    def __repr__(self):
        return 'torch.cat(({{}}), dim={})'.format(self.dim)

class Sum(PlaceHolder):
    """A class that represents the torch elementwise addition operation
    between tensors"""

    def __repr__(self):
        return 'torch.add({0}, 1, {1})'

class Flatten(PlaceHolder):
    """A class that represents the torch tesnor.view() operation"""

    def __init__(self, **kwargs):
        pass # void calling super

    def __repr__(self):
        return '{0}.view({0}.size(0), -1)'

class AffineGridGen(PlaceHolder):
    """A class that represents the torch affine_grid function

    Note that the ordering of the affine parameter matrix differs between
    pytorch and mcn.  Consequently, we must re-order the elements (see the
    __repr__ below for the ordering).

    The interface for this function is slightly unusual - it takes a `size`
    argument in the form NxCxHxW (e.g. torch.Size((1,3,100,100))).  However,
    the channel component C is not used by the function, and is non-trivial
    to determine from the matconvnet graph (which does not store this
    information. To address this issue, we perform an 'interface alignment'
    step on the network to treat the feature map that the grid will be applied
    to as an extra input to this layer.
    """
    def __init__(self, height, width, dummy_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.size = (1, dummy_channels, height, width)

    def __repr__(self):
        return ('F.affine_grid({0}.view(-1,6)[:,[3, 2, 5, 1, 0, 4]]'
                '.view(-1, 2, 3), {1}.size())')

class BilinearSampler(PlaceHolder):
    """A class that represents the torch (bilinear) grid_sample function
    """

    def __repr__(self):
        return "F.grid_sample({0}, {1}, mode='bilinear')"

#class GlobalPooling(PlaceHolder):
#    """A class that represents the Global Pooling 2d operation"""
#
#    def __init__(self, pad=0, dilate=(1,1), method='avg', stride=(1, 1),
#                 **kwargs):
#        if isinstance(pad, np.ndarray) and pad.sum() == 0:
#            pad = 0 # use common format
#        method = method.lower()
#        assert method in ['avg', 'max'], 'unrecognised pooling method'
#        if isinstance(stride, np.ndarray) and (stride == 1).all():
#            stride = 1
#        super().__init__(**kwargs)
#        self.pad = pad
#        self.stride = stride
#        self.method = method
#
#    def __repr__(self):
#        return "nn.{}_pool2d({{0}}, {{0}}.size()[2:], stride={}, padding={})"\
#                .format(self.method, self.stride, self.pad)

class Permute(PlaceHolder):
    """A class that represents the torch.tranpose() operation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = self.block['order'].flatten() - 1 # fix 1-indexing

    def __repr__(self):
        import ipdb ; ipdb.set_trace()
        #TODO(samuel): add logic for higher dims
        changes = self.order - np.arange(4)
        error_msg = 'only two dims can be transposed at a time'
        assert (changes == 0).sum() <= 2, error_msg
        error_msg = 'only tranpose along first dimensions currently supported'
        assert np.array_equal(np.where(changes != 0)[0], [0, 1]), error_msg
        dim0 = np.where(self.order == 0)[0][0]
        dim1 = np.where(self.order == 1)[0][0]
        return 'torch.tranpose({{}}, {}, {})'.format(dim0, dim1)

class Reshape(PlaceHolder):
    """A class that represents the torch.view() operation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = self.block['shape']
        if (shape == 0).all(): # if shape is an array of zeroes, no reshape occurs
            self.blank = True # ensure that no output is produced
        else:
            import ipdb ; ipdb.set_trace()

    def __repr__(self):
        import ipdb ; ipdb.set_trace()
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
        if x.shape[:2] == (1,1): # linear layers
            x = x[0,0,:,:].T
        else:
            x = x.transpose((3, 2, 0, 1))
    elif x.ndim == 2:
        if x.shape[1] == 1:
            x = x.flatten()
    return torch.from_numpy(x)

def capitalize_first_letter(s):
    """Capitalize (only) the first character of a string

    This is an alternative to the  built-in str function capitalize()
    which changes all letters following the first to lower case.

    Args:
        s (str): the string to be capitalized.

    Returns:
        (str): the modified string
    """
    base = []
    if len(s) > 0:
        base += s[0].upper()
    if len(s) > 1:
        base += s[1:]
    return ''.join(base)

def lower_first_letter(s):
    """Lower (only) the first character of a string

    Args:
        s (str): the string to be modified.

    Returns:
        (str): the modified, lowercase prefixed string
    """
    base = []
    if len(s) > 0:
        base += s[0].lower()
    if len(s) > 1:
        base += s[1:]
    return ''.join(base)


def merge_two_dicts(a, b):
    """Merge the contents of two dictionaries into a single dictionary.

    NOTE: This function is primarily used to help with python2 compatibility
    during argument parsing"""
    z = a.copy()
    z.update(b)
    return z
