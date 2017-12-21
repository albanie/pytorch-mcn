# -*- coding: utf-8 -*-
"""Import script
"""

from os.path import join as pjoin
import numpy as np
import scipy.io as sio

mcn_dir = '/users/albanie/data/models/matconvnet'
model_name = 'squeezenet1_0-pt-mcn.mat'
ref_path = pjoin(mcn_dir, model_name)
print('loading mcn model...')
mcn = sio.loadmat(ref_path)


def parse_struct(x):
    """Extract nested dict data structure from matfile struct layout

    Args:
        x (ndarray): a nested, mixed type numpy array that has been
                     loaded from a .mat file with the scipy.io.loadmat
                     utility .

    Returns:
        nested dictionary of parsed data
    """
    # handle scalar base cases for recursion
    scalar_types = [np.str_, np.uint8, np.int64, np.float32, np.float64]
    if x.dtype.type in scalar_types:
        return x
    fieldnames = [tup[0] for tup in x.dtype.descr]
    parsed = {f:[] for f in fieldnames}
    if any([dim == 0 for dim in x.shape]):
        return parsed # only recurse on valid storage shapes
    if fieldnames == ['']:
        return x[0][0] # prevent blank nesting
    for f_idx, fname in enumerate(fieldnames):
        x = x.flatten() # simplify logic via linear index
        for ii in range(x.size):
            parsed[fname].append(parse_struct(x[ii][f_idx]))
    return parsed

# sanity check
for key in ['meta', 'params', 'layers']:
    assert key in mcn.keys()

# meta_dict = parse_struct(mcn['meta'])
# params_dict = parse_struct(mcn['params'])
layers_dict = parse_struct(mcn['layers'])
