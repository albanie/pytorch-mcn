# -*- coding: utf-8 -*-
"""A collection of functions for generating PyTorch model
definition source code.

-----------------------------------------------------------
Licensed under The MIT License [see LICENSE.md for details]
Copyright (C) 2017 Samuel Albanie
-----------------------------------------------------------
"""

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
