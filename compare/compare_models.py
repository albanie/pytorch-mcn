# -*- coding: utf-8 -*-
"""Feature comparison

This module demonstrates intermediate feature comparison, which can be used
as a sanity check when importing models from MatConvNet into PyTorch.
"""

import torch
from torch.autograd import Variable
import scipy.io as sio
from models.squeezenet1_0 import squeezenet1_0

def array2var(a):
    """transpose array (which have been stored in column-major
    order) to fit the expected PyTorch tensor shape and load as
    an Autograd Variable

    Args:
        a (ndarray): A numpy array of matconvnet features, stored in
        HxWxCxN layout
    Returns:
        (autograd.Variable): a variable containing the data in PyTorch layout
    """
    if a.ndim == 3:
        a = a.transpose((2, 0, 1))
    return Variable(torch.from_numpy(a).unsqueeze(0))

def show_diffs(mcn_vars, py_vars):
    """display differences between intermediate variables of both networks

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
    """compare features computed by networks in each framework

    Args:
        net (nn.Module): The imported PyTorch network
        mcn_feat_path (str): path to the location of the saved intermediate
            matconvent features
    """
    mcn_vars = sio.loadmat(mcn_feat_path)
    data = mcn_vars['data']
    data = array2var(data)
    net.eval()
    net.forward_debug(data)
    py_vars = net.debug_feats
    show_diffs(mcn_vars, py_vars)

weights_path = 'weights/squeezenet1_0.pth'
mcn_feat_path = 'feats/squeezenet1_0-pt-mcn-feats.mat'
net = squeezenet1_0(weights_path=weights_path)
compare_network_features(net, mcn_feat_path)
