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

# sanity check
for key in ['meta', 'params', 'layers']:
    assert key in mcn.keys()

def parse_struct(x):
    """extract nested dict from matfile struct layout
    """
    # handle base case
    scalar_types = [np.str_, np.uint8, np.float32, np.float64]
    if x.dtype.type in scalar_types:
        return x
        # import ipdb ; ipdb.set_trace()
    fieldnames = [tup[0] for tup in x.dtype.descr]
    d = {f:[] for f in fieldnames}
    # only recurse on valid storage shapes
    if any([dim == 0 for dim in x.shape]):
        return d
    # if '' in fieldnames: import ipdb ; ipdb.set_trace()
    # print(fieldnames)
    for f_idx, fname in enumerate(fieldnames):
        # it = np.nditer(x, flags=['f_index', 'refs_ok', 'external_loop'])
        # while not it.finished:
            # elem = it[0]
            # # (it[0], it.index)
            # import ipdb ; ipdb.set_trace()
            # d[fname].append(parse_struct(elem[it.index][f_idx]))
            # d[fname].append(parse_struct(elem[it.index][f_idx]))
        # while 
            # d[fname].append(parse_struct(elem[0][idx]))
        x = x.flatten() # seems safer the external looping over refs
        for ii in range(x.size):
        # for elem in np.nditer(x, order='F', flags=('refs_ok','external_loop')):
            # import ipdb ; ipdb.set_trace()
            d[fname].append(parse_struct(x[ii][f_idx]))
    return d

# meta_dict = parse_struct(mcn['meta'])
params_dict = parse_struct(mcn['params'])
