# -*- coding: utf-8 -*-
"""This module evaluates imported PyTorch models
"""

import os
import argparse
import importlib.util
from imagenet import imagenet_benchmark
from torchvision.models import densenet

# directory containing imported pytorch models
MODEL_DIR = os.path.expanduser('~/data/models/pytorch/mcn_imports/')

# imagenet directory
ILSVRC_DIR = os.path.expanduser('~/data/datasets/ILSVRC2012')

# results cache directory
CACHE_DIR = '../res_cache'

def load_torchvision_model(model_name):
    if 'densenet' in model_name:
        func = getattr(densenet, model_name)
        net = func(pretrained=True)
        net.meta = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'imageSize': [224, 224]}
    return net

def load_model(model_name):
    """Load imoprted PyTorch model by name

    Args:
        model_name (str): the name of the model to be loaded

    Return:
        nn.Module: the loaded network
    """
    if 'tv_' in model_name:
        import ipdb ; ipdb.set_trace()
        net = load_torchvision_model(model_name)
    else:
        model_def_path = os.path.join(MODEL_DIR, model_name +'.py')
        weights_path = os.path.join(MODEL_DIR, model_name + '.pth')
        spec = importlib.util.spec_from_file_location(model_name, model_def_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        func = getattr(mod, model_name)
        net = func(weights_path=weights_path)
    return net

def run_benchmarks(gpus, refresh, remove_blacklist):
    """Run bencmarks for imported models

    Args:
        gpus (str): comma separated gpu device identifiers
        refresh (bool): whether to overwrite the results of existing runs
        remove_blacklist (bool): whether to remove images from the 2014 ILSVRC
          blacklist from the validation images used in the benchmark
    """

    # Select models (and their batch sizes) to include in the benchmark.
    model_list = [
        ('alexnet_pt_mcn', 256),
        ('squeezenet1_0_pt_mcn', 128),
        ('squeezenet1_1_pt_mcn', 128),
        ('vgg11_pt_mcn', 128),
        ('vgg13_pt_mcn', 92),
        ('vgg16_pt_mcn', 32),
        ('vgg19_pt_mcn', 24),
        ('resnet18_pt_mcn', 50),
        ('resnet34_pt_mcn', 50),
        ('resnet50_pt_mcn', 32),
        ('resnet101_pt_mcn', 24),
        ('resnet152_pt_mcn', 20),
        ('inception_v3_pt_mcn', 64),
        ("densenet121_pt_mcn", 50),
        ("densenet161_pt_mcn", 32),
        ("densenet169_pt_mcn", 32),
        ("densenet201_pt_mcn", 32),
        ("tv_densenet121", 32),
        ('imagenet_matconvnet_alex', 256),
        ('imagenet_matconvnet_vgg_f_dag', 128),
        ('imagenet_matconvnet_vgg_m_dag', 128),
        ('imagenet_matconvnet_vgg_verydeep_16_dag', 32),
    ]

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)

    opts = {'data_dir': ILSVRC_DIR, 'refresh_cache': refresh,
            'remove_blacklist': remove_blacklist}

    for model_name, batch_size in model_list:
        opts['res_cache'] = '{}/{}.pth'.format(CACHE_DIR, model_name)
        model = load_model(model_name)
        imagenet_benchmark(model, batch_size=batch_size, **opts)

parser = argparse.ArgumentParser(description='Run PyTorch benchmarks.')
parser.add_argument('--gpus', nargs='?', dest='gpus',
                    help='select gpu device id')
parser.add_argument('--refresh', dest='refresh', action='store_true',
                    help='refresh results cache')
parser.add_argument('--remove-blacklist', dest='remove_blacklist',
                    action='store_true',
                    help=('evaluate on 2012 validation subset without including'
                    'the 2014 list of blacklisted images'))
parser.set_defaults(gpus=None)
parser.set_defaults(refresh=False)
parser.set_defaults(remove_blacklist=False)
parsed = parser.parse_args()

if __name__ == '__main__':
    run_benchmarks(parsed.gpus, parsed.refresh, parsed.remove_blacklist)
