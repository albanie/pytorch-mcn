# -*- coding: utf-8 -*-
"""Utilties shared among the benchmarking protocols
"""
import torchvision.transforms as transforms

def compose_transforms(meta):
    """Compose preprocessing transforms for model

    The imported models use a range of different preprocessing options,
    depending on how they were originally trained. Models trained in MatConvNet
    typically require input images that have been scaled to [0,255], rather
    than the [0,1] range favoured by PyTorch.

    Args:
        meta (dict): model preprocessing requirements

    Return:
        (transforms.Compose): Composition of preprocessing transforms
    """
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    transform_list = [transforms.Resize((im_size[0], im_size[1])),
                      transforms.ToTensor()]
    if meta['std'] == [1,1,1]: # common amongst mcn models
        transform_list.append(lambda x: x * 255.0)
    transform_list.append(normalize)
    return transforms.Compose(transform_list)
