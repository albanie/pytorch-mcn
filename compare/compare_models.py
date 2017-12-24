# -*- coding: utf-8 -*-
"""Feature comparison

This module demonstrates intermediate feature comparison, which can be used
as a sanity check when importing models from MatConvNet into PyTorch.
"""

import os
import argparse
import importlib.util
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
    return Variable(torch.from_numpy(array).unsqueeze(0))

def show_diffs(mcn_vars, py_vars):
    """Display differences between intermediate variables of both networks.

    Args:
        mcn_vars (dict): features computed from the matconvnet network
        py_vars (dict): features computed from the PyTorch network
    """
    for varname in py_vars.keys():
        x = py_vars[varname].contiguous()
        x_ = array2var(mcn_vars[varname]).contiguous()
        x = x.view(x.size(0), -1)
        x_ = x_.view(x_.size(0), -1)
        diff = torch.sum(torch.abs(x) - torch.abs(x_))/torch.sum(torch.abs(x))
        print('{} diff: {:.3g}'.format(varname, diff.data[0]))

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
    data = mcn_vars['data']
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

# import module
spec = importlib.util.spec_from_file_location(model_name, model_def_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
func = getattr(mod, model_name)
net = func(weights_path=weights_path)

# verify features
compare_network_features(net, mcn_feat_path)
