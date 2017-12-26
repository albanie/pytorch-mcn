# -*- coding: utf-8 -*-
"""This module evaluates imported PyTorch models
"""

import os
import argparse
import importlib.util
from imagenet import imagenet_benchmark

# directory containing imported pytorch models
MODEL_DIR = os.path.expanduser('~/data/models/pytorch/mcn_imports/')

# imagenet directory
ILSVRC_DIR = os.path.expanduser('~/data/datasets/ILSVRC2012')

# results cache directory
CACHE_DIR = '../res_cache'

def load_model(model_name):
    """Load imoprted PyTorch model by name

    Args:
        model_name (str): the name of the model to be loaded
    """
    model_def_path = os.path.join(MODEL_DIR, model_name +'.py')
    weights_path = os.path.join(MODEL_DIR, model_name + '.pth')
    spec = importlib.util.spec_from_file_location(model_name, model_def_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    func = getattr(mod, model_name)
    net = func(weights_path=weights_path)
    return net

def run_benchmarks(gpus, refresh):

    model_list = [
        ('squeezenet1_0_pt_mcn', 128),
        ('squeezenet1_1_pt_mcn', 128),
    ]

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)

    opts = {'data_dir': ILSVRC_DIR, 'refresh_cache': refresh}

    for model_name, batch_size in model_list:
        opts['res_cache'] = '{}/{}.pth'.format(CACHE_DIR, model_name)
        model = load_model(model_name)
        imagenet_benchmark(model, batch_size=batch_size, **opts)

parser = argparse.ArgumentParser(description='Run PyTorch benchmarks.')
parser.add_argument('--gpus', nargs='?', dest='gpus',
                    help='select gpu device id')
parser.add_argument('--refresh', dest='refresh', action='store_true',
                    help='refresh results cache')
parser.set_defaults(gpus=None)
parser.set_defaults(refresh=False)
parsed = parser.parse_args()

if __name__ == '__main__':
    run_benchmarks(parsed.gpus, parsed.refresh)
