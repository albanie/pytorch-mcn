# -*- coding: utf-8 -*-
"""LFW benchmark for face verification. This is designed to be used as a
sanity check for imported models.

This code is primarily based on the code of https://github.com/clcarwin. The
original code can be found here:
https://github.com/clcarwin/sphereface_pytorch

License from original codebase:

MIT License

Copyright (c) 2017 carwin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import torch
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

from zsvision.zs_iterm import zs_dispFig

from PIL import Image
import benchmark_utils

import tqdm
import cv2
import argparse
import numpy as np
import zipfile

import sys
sys.path.insert(0, '/users/albanie/coding/libs/pt/sphereface_pytorch')

from matlab_cp2tform import get_similarity_transform_for_cv2
# import net_sphere

def alignment(src_img, src_pts, output_size=(96, 112)):
    """Warp a face image so that its features align with a canoncial
    reference set of landmarks. The alignment is performed with an
    affine warp

    Args:
        src_img (ndarray): an HxWx3 RGB containing a face
        src_pts (ndarray): a 5x2 array of landmark locations
        output_size (tuple): the dimensions (oH, oW) of the output image

    Returns:
        (ndarray): an (oH x oW x 3) warped RGB image.
    """
    ref_pts = [[30.2946, 51.6963],
               [65.5318, 51.5014],
               [48.0252, 71.7366],
               [33.5493, 92.3655],
               [62.7299, 92.2041]]
    src_pts = np.array(src_pts).reshape(5,2)
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)
    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, output_size)
    return face_img


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i*n/n_folds:(i+1)*n/n_folds]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

# directory containing imported pytorch models
import os, six
MODEL_DIR = os.path.expanduser('~/data/models/pytorch/mcn_imports/')

def load_model(model_name):
    """Load imoprted PyTorch model by name

    Args:
        model_name (str): the name of the model to be loaded

    Return:
        nn.Module: the loaded network
    """
    model_def_path = os.path.join(MODEL_DIR, model_name +'.py')
    weights_path = os.path.join(MODEL_DIR, model_name + '.pth')
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
    return net


parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--lfw', default='data/lfw.zip', type=str)
parser.add_argument('--model_name', default='resnet50_scratch_dag', type=str)
args = parser.parse_args()

predicts=[]
# net = getattr(net_sphere,args.net)()
net = load_model(args.model_name)
# net.load_state_dict(torch.load(args.model))
net.cuda()
net.eval()
net.feature = True

zfile = zipfile.ZipFile(args.lfw)

landmark = {}
with open('data/lfw_landmark.txt') as f:
    landmark_lines = f.readlines()
for line in landmark_lines:
    l = line.replace('\n','').split('\t')
    landmark[l[0]] = [int(k) for k in l[1:]]

with open('data/pairs.txt') as f:
    pairs_lines = f.readlines()[1:]

from skimage.transform import resize

for i in tqdm.tqdm(range(6000)):
# for i in tqdm.tqdm(range(600)):
    p = pairs_lines[i].replace('\n','').split('\t')

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))

    im1 = cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1)
    img1_aligned = alignment(im1, landmark[name1])
    im2 = cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1)
    img2_aligned = alignment(im2, landmark[name2])

    # convert images to PIL to use builtin transforms
    import matplotlib.pyplot as plt
    img1 = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2RGB)
    img1 = Image.fromarray(img1)
    img2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(img2)

    # debug
    # plt.imshow(img1_aligned) ; zs_dispFig()
    # plt.imshow(img1) ; zs_dispFig()
    # import ipdb ; ipdb.set_trace()

    meta = net.meta
    preproc_transforms = benchmark_utils.compose_transforms(meta)

    # imglist = [img1, cv2.flip(img1, 1), img2, cv2.flip(img2, 1)]
    imglist = [img1, img1.transpose(Image.FLIP_LEFT_RIGHT),
              img2, img2.transpose(Image.FLIP_LEFT_RIGHT)]

    for i in range(len(imglist)):
        imglist[i] = preproc_transforms(imglist[i])
        # vis
        # plt.imshow(imglist[0].numpy().transpose(1,2,0))
        # import ipdb ; ipdb.set_trace()
        #imglist[i] = resize(imglist[i], (224,224), mode='reflect')

        ## imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
        #imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,224,224))
        #imglist[i] = (imglist[i]-127.5)/128.0

    # img = np.vstack(imglist)
    imgs = torch.stack(imglist, dim=0)
    imgs = Variable(imgs).cuda()
    # img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    _, output = net(imgs)
    # output = net(img)
    f = output.data
    f1,f2 = f[0].squeeze(),f[2].squeeze()
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))


accuracy = []
thd = []
folds = KFold(n=6000, n_folds=10, shuffle=False)
# folds = KFold(n=600, n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.005)
predicts = np.array(map(lambda line:line.strip('\n').split(), predicts))
for idx, (train, test) in tqdm.tqdm(enumerate(folds)):
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
