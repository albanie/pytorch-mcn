# -*- coding: utf-8 -*-
"""Feature comparison

This module demonstrates intermediate feature comparison, which can be used
as a sanity check when importing models from MatConvNet into PyTorch.
"""

import os
import six
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import scipy.io as sio

def array2var(array):
    """Transpose an array (stored in column-major order) to fit the expected
    PyTorch tensor shape and load as an Autograd Variable.

    Args:
        a (ndarray): A numpy array of matconvnet features, stored in
        HxWxCxN layout
    Returns:
        (autograd.Variable): a variable containing the data in PyTorch layout
    """
    if array.ndim == 3:
        array = array.transpose((2, 0, 1))
    var = Variable(torch.from_numpy(array))
    while len(var.size()) < 4:
        var = var.unsqueeze(0)
    return var

def show_diffs(mcn_vars, py_vars):
    """Display differences between intermediate variables of both networks.

    When converting models from MatConvNet to PyTorch, we often insert an
    additional view operation to ensure output shape consistency. Since this
    operation is not present in the corresponding mcn network, it is skipped
    in the comparison.

    The suffix `_preflatten` is used to denote the output variable of the
    module before the View operation.

    Args:
        mcn_vars (dict): features computed from the matconvnet network
        py_vars (dict): features computed from the PyTorch network
    """
    skip = False
    for idx, varname in enumerate(py_vars.keys()):
        if skip:
            skip = False
            continue
        x = py_vars[varname].contiguous()
        if '_preflatten' in varname:
            varname = varname[:varname.index('_preflatten')]
            skip = True
        mcn_var = mcn_vars[varname]
        if 'grid' in varname and mcn_var.shape[0] == 2:
            # affine grids are ordered differently
            mcn_var = np.expand_dims(mcn_var.transpose((1, 2, 0)), 0)
        x_ = array2var(mcn_var).contiguous()
        x = x.view(x.size(0), -1)
        x_ = x_.view(x_.size(0), -1)
        diff = torch.sum(torch.abs(x) - torch.abs(x_))/torch.sum(torch.abs(x))
        print('{} diff: {:.3g}'.format(varname, diff.item()))

def compare_network_features(net, mcn_feat_path):
    """Compare features computed by networks in each framework.

    Args:
        net (nn.Module): The imported PyTorch network
        mcn_feat_path (str): path to the location of the saved intermediate
            matconvent features
    """
    msg = ('The intermediate features of the MatConvNet model (required for '
           'verifcation) were not found, did you run the featureDumper.m '
           'MATLAB script for the target model?')
    assert os.path.isfile(mcn_feat_path), msg
    mcn_vars = sio.loadmat(mcn_feat_path)
    in_var_names = ['data', 'input', 'x0']
    msg = 'could not find unique input variable'
    assert sum([x in mcn_vars for x in in_var_names]) == 1, msg
    for var_name in in_var_names:
        if var_name in mcn_vars:
            data = mcn_vars[var_name]
    data = array2var(data)
    net.eval()
    net.forward_debug(data)
    py_vars = net.debug_feats
    show_diffs(mcn_vars, py_vars)

parser = argparse.ArgumentParser(
    description='Compare features from MatConvNet and PyTorch models.')
parser.add_argument('mcn_model_path', type=str,
                    help='The input should be the path to a matconvnet model \
                    file (a .mat) file, stored in either dagnn or simplenn \
                    format')
parser.add_argument('output_dir', type=str,
                    help='location of imported pytorch models')
parser.add_argument('feat_dir', type=str,
                    help='location of stored MatConvNet model features')
parsed = parser.parse_args()

mcn_model_path = os.path.expanduser(parsed.mcn_model_path)
output_dir = os.path.expanduser(parsed.output_dir)
feat_dir = os.path.expanduser(parsed.feat_dir)

# configure paths
model_name = os.path.basename(os.path.splitext(mcn_model_path)[0])
model_name = model_name.replace('-', '_') # sanitize model name for python
mcn_feat_path = os.path.join(feat_dir, '{}-feats.mat'.format(model_name))
model_def_path = os.path.join(output_dir, model_name +'.py')
weights_path = os.path.join(output_dir, model_name + '.pth')

# import module (uses different approaches for Python2/3)
if six.PY3:
    import importlib.util
    spec = importlib.util.spec_from_file_location(model_name, model_def_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
else:
    import importlib, sys
    dirname = os.path.dirname(model_def_path)
    sys.path.insert(0, dirname)
    module_name = os.path.splitext(os.path.basename(model_def_path))[0]
    mod = importlib.import_module(module_name)

func = getattr(mod, model_name)
net = func(weights_path=weights_path)
compare_network_features(net, mcn_feat_path) # verify features
