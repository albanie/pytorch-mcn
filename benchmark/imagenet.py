# -*- coding: utf-8 -*-
"""Imagenet validation set benchmark

The module evaluates the performance of a pytorch model on the ILSVRC 2012
validation set.

Based on PyTorch imagenet example:
    https://github.com/pytorch/examples/tree/master/imagenet
"""

import os
import time

from PIL import ImageFile
import torch
import torch.nn.parallel
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

def imagenet_benchmark(model, data_dir, res_cache, refresh_cache,
                       batch_size=256, num_workers=20):
    if not refresh_cache: # load result from cache, if available
        if os.path.isfile(res_cache):
            res = torch.load(res_cache)
            prec1, prec5, speed = res['prec1'], res['prec5'], res['speed']
            print("=> loaded results from '{}'".format(res_cache))
            info = (100 - prec1, 100 - prec5, speed)
            msg = 'Top 1 err: {:.2f}, Top 5 err: {:.2f}, Speed: {:.1f}Hz'
            print(msg.format(*info))
            return

    meta = model.meta
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model).cuda()
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=meta['mean'], std=meta['std'])
    im_size = meta['imageSize']
    assert im_size[0] == im_size[1], 'expected square image size'
    crop_size = int((256/224) * im_size[0])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(im_size[0]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    prec1, prec5, speed = validate(val_loader, model)
    torch.save({'prec1': prec1, 'prec5': prec5, 'speed': speed}, res_cache)

def validate(val_loader, model):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    speed = WarmupAverageMeter()
    end = time.time()
    for ii, (ims, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        ims_var = torch.autograd.Variable(ims, volatile=True)
        output = model(ims_var) # compute output
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1[0], ims.size(0))
        top5.update(prec5[0], ims.size(0))
        speed.update(ims.size(0)/(time.time() - end))
        end = time.time()
        if ii % 10 == 0:
            print('Test: [{0}/{1}]\t' 'Speed {speed.val:.1f}Hz ({speed.avg:.1f})Hz '
                  'Prec@1 {top1.avg:.3f} {top5.avg:.3f}'.format(
                      ii, len(val_loader), speed=speed, top1=top1, top5=top5))
    top1_err, top5_err = 100 - top1.avg, 100 - top5.avg
    print(' * Err@1 {0:.3f} Err@5 {1:.3f}'.format(top1_err, top5_err))

    return top1.avg, top5.avg, speed.avg

class WarmupAverageMeter(object):
    """Computes and stores the average and current value, after a fixed
    warmup period (useful for approximate benchmarking)

    Args:
        warmup (int) : The number of updates to be ignored before the average
        starts to be computed.
    """
    def __init__(self, warmup=5):
        self.reset()
        self.warmup = warmup

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.warmup_count = 0

    def update(self, val, n=1):
        self.warmup_count = self.warmup_count + 1
        if self.warmup_count >= self.warmup:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
