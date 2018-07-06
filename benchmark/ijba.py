# -*- coding: utf-8 -*-
"""IJB-A benchmark for face verification

This code is primarily based on the code of Aruni Roy Chowdhury. The original
code can be found here:
https://github.com/AruniRC/resnet-face-pytorch/blob/master/ijba/eval_ijba_1_1.py
"""

import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
import benchmark_utils

import yaml
import tqdm
import numpy as np
import sklearn.metrics
from sklearn import metrics
from scipy import interpolate
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#here = osp.dirname(osp.abspath(__file__)) # output folder is located here
#root_dir,_ = osp.split(here)
#import sys
#sys.path.append(root_dir)

# for debugging purposes
import sys
sys.path.insert(0, '/users/albanie/coding/libs/pt/resnet-face-pytorch')

import models
import ijba_utils.misc
import data_loader


'''
Evaluate a network on the IJB-A 1:1 verification task
=====================================================
Example usage: TODO ***
# Resnet 101 on 10 folds of IJB-A 1:1
'''
# MODEL_PATH = '/srv/data1/arunirc/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_vggface_scratch_CFG-022_TIME-20180210-201442/model_best.pth.tar'


# Resnet101-512d-norm
# MODEL_PATH = '/home/renyi/arunirc/data1/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_512d_L2norm_ft2_CFG-023_TIME-20180214-020054/model_best.pth.tar'
MODEL_TYPE = 'resnet101-512d-norm'
# MODEL_PATH = '/home/renyi/arunirc/data1/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_512d_L2norm_ft2_CFG-022_TIME-20180214-015313/model_best.pth.tar'
MODEL_PATH = '/home/renyi/arunirc/data1/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_512d_L2norm_ft2_CFG-024_TIME-20180214-160410/model_best.pth.tar'


# Resnet101-512d
# MODEL_PATH = '/home/renyi/arunirc/data1/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_bottleneck_ft2_CFG-023_TIME-20180213-091016/model_best.pth.tar'
# MODEL_TYPE = 'resnet101-512d'
# MODEL_PATH = '/home/renyi/arunirc/data1/Research/resnet-face-pytorch/vgg-face-2/logs/MODEL-resnet101_bottleneck_ft1_CFG-021_TIME-20180212-192332/model_best.pth.tar'

MODEL_TYPES = ['resnet50', 'resnet101', 'resnet101-512d', 'resnet101-512d-norm']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='ijba_eval')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir',
                        default='/scratch/local/ramdisk/albanie/ijba/cropped')
    parser.add_argument('--protocol_dir',
                default='/users/albanie/data/datasets/ijba/protocol/IJB-A_11')
    parser.add_argument('--fold', type=int, default=1, choices=[1,10])
    parser.add_argument('--cache_dir',
               default='/users/albanie/coding/libs/pt/pytorch-mcn/ijba-feats')
    parser.add_argument('--sqrt', action='store_true', default=False,
                        help='Add signed sqrt normalization')
    parser.add_argument('--cosine', action='store_true', default=False,
                        help='Use cosine similarity instead of L2 distance')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--refresh', type=bool, default=False)
    parser.add_argument('--model_dir',
                       help='directory containing pretrained model')
    parser.add_argument('--model_name', default='resnet50_scratch_dag',
                     help=('name of the pretrained model (both the model and'
                     'its weights are expected to have the same filename, but'
                     'different file extensions e.g `model.py` and `model.pth`'
                     'for the model definition and its weights respectively'))

    #parser.add_argument('-m', '--model_path',
    #                    default=MODEL_PATH,
    #                    help='Path to pre-trained model')
    #parser.add_argument('--model_type', default=MODEL_TYPE, choices=MODEL_TYPES)
    args = parser.parse_args()


    # CUDA setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True # enable if all images are same size


    # -----------------------------------------------------------------------------
    # 1. Model
    # -----------------------------------------------------------------------------
    num_class = 8631 # number of identities in VGG-Face2
    #if args.model_type == 'resnet50':
    #    model = torchvision.models.resnet50(pretrained=False)
    #    model.fc = torch.nn.Linear(2048, num_class)
    #elif args.model_type == 'resnet101':
    #    model = torchvision.models.resnet101(pretrained=False)
    #    model.fc = torch.nn.Linear(2048, num_class)
    #elif args.model_type == 'resnet101-512d':
    #    model = torchvision.models.resnet101(pretrained=False)
    #    layers = []
    #    layers.append(torch.nn.Linear(2048, 512))
    #    layers.append(torch.nn.Linear(512, num_class))
    #    model.fc = torch.nn.Sequential(*layers)
    #elif args.model_type == 'resnet101-512d-norm':
    #    model = torchvision.models.resnet101(pretrained=False)
    #    layers = []
    #    layers.append(torch.nn.Linear(2048, 512))
    #    layers.append(models.NormFeat(scale_factor=50.0))
    #    layers.append(torch.nn.Linear(512, num_class))
    #    model.fc = torch.nn.Sequential(*layers)
    #else:
    #    raise NotImplementedError


    model_def_path = os.path.join(args.model_dir, args.model_name +'.py')
    weights_path = os.path.join(args.model_dir, args.model_name + '.pth')
    import importlib, sys
    dirname = os.path.dirname(model_def_path)
    sys.path.insert(0, dirname)
    module_name = os.path.splitext(os.path.basename(model_def_path))[0]
    mod = importlib.import_module(module_name)
    func = getattr(mod, args.model_name)
    model = func(weights_path=weights_path)

    #import_dir = '/scratch/shared/nfs1/albanie/models/pytorch/mcn_imports'
    #sys.path.insert(0, import_dir)
    #from resnet50_scratch_dag import Resnet50_scratch_dag

    #checkpoint = torch.load(args.model_path)

    #if checkpoint['arch'] == 'DataParallel':
    #    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4])
    #    model.load_state_dict(checkpoint['model_state_dict'])
    #    model = model.module # get network module from inside its DataParallel wrapper
    #else:
    #    model.load_state_dict(checkpoint['model_state_dict'])

    if cuda:
        model = model.cuda()

    # Convert the trained network into a "feature extractor" by removing the
    # classifier
    #feature_map = list(model.children())
    #feature_map.pop() # pop the classifier
    #extractor = nn.Sequential(*feature_map)
    extractor = model
    extractor.eval() # set to evaluation mode (fixes BatchNorm, dropout, etc.)

    #if args.model_type == 'resnet101-512d' or args.model_type == 'resnet101-512d-norm':
    #    model.eval()
    #    extractor = model
    #    extractor.fc = nn.Sequential(extractor.fc[0])
    #else:
    #    extractor = nn.Sequential(*feature_map)



    # -----------------------------------------------------------------------------
    # 2. Dataset
    # -----------------------------------------------------------------------------
    fold_id = 1
    file_ext = '.jpg'
    #RGB_MEAN = [ 0.485, 0.456, 0.406 ]
    #RGB_STD = [ 0.229, 0.224, 0.225 ]

    preproc_transforms = benchmark_utils.compose_transforms(model.meta)
    #test_transform = transforms.Compose([
    #    # transforms.Scale(224),
    #    # transforms.CenterCrop(224),
    #    transforms.Scale((224,224)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean = RGB_MEAN,
    #                         std = RGB_STD),
    #])

    pairs_path = osp.join(args.protocol_dir, 'split{}'.format(fold_id),
                          'verify_comparisons_{}.csv'.format(fold_id))
    pairs = ijba_utils.misc.read_ijba_pairs(pairs_path)
    protocol_file = osp.join(args.protocol_dir, 'split%d' % fold_id,
                          'verify_metadata_%d.csv' % fold_id)
    metadata = ijba_utils.misc.get_ijba_1_1_metadata(protocol_file) # dict
    assert np.all(np.unique(pairs) == np.unique(metadata['template_id']))  # sanity-check

    # face crops saved as <sighting_id.jpg>
    path_list = np.array([osp.join(args.data_dir, str(x)+file_ext)
                        for x in metadata['sighting_id'] ])
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
                        data_loader.IJBADataset(
                        path_list, preproc_transforms, split=fold_id),
                        batch_size=args.batch_size, shuffle=False )

    # testing
    # for i in range(len(test_loader.dataset)):
    #     img = test_loader.dataset.__getitem__(i)
    #     sz = img.shape
    #     if sz[0] != 3:
    #         print sz

    # -----------------------------------------------------------------------------
    # 3. Feature extraction
    # -----------------------------------------------------------------------------
    # This does twice as much work as needed. Can fix later.
    print('Feature extraction...')
    cache_dir = osp.join(args.cache_dir, 'cache-' + args.model_name)
    if not osp.exists(cache_dir):
        os.makedirs(cache_dir)

    feat_path = osp.join(cache_dir, 'feat-fold-{}.mat'.format(fold_id))

    if not osp.exists(feat_path) or args.refresh:
        features = []
        for batch_idx, images in tqdm.tqdm(enumerate(test_loader),
                                           total=len(test_loader),
                                           desc='Extracting features'):
            x = Variable(images, volatile=True) # test-time memory conservation
            if cuda:
                x = x.cuda()
            scores, feat = extractor(x)
            if cuda:
                feat = feat.data.cpu() # free up GPU
            else:
                feat = feat.data
            features.append(feat)

        features = torch.cat(features, dim=0) # (n_batch*batch_sz) x 512
        sio.savemat(feat_path, {'feat': features.cpu().numpy() })
    else:
        dat = sio.loadmat(feat_path)
        features = torch.FloatTensor(dat['feat'])
        del dat
        print('Loaded.')


    # -----------------------------------------------------------------------------
    # 4. Verification
    # -----------------------------------------------------------------------------
    def compute_pair_labels(pairs, metadata):
        """Compute labels for each pair of verification comparisons. For
        a given pair, the label is defined to be `1` if both identities
        are the same and `0` otherwise.

        Args:
            pairs (ndarray): Nx2 array containing the indices of each test
              pair, where N is the total number of pairs.
            metadata (dict): all metadata associated with the verification
              evaluation.

        Returns:
            (ndarray): 1xN an array of pair labels.
        """
        print('computing pair labels...')
        labels = []
        for pair in tqdm.tqdm(pairs):
            sel_t0 = np.where(metadata['template_id'] == pair[0])[0]
            sel_t1 = np.where(metadata['template_id'] == pair[1])[0]
            subject0 = np.unique(metadata['subject_id'][sel_t0])
            subject1 = np.unique(metadata['subject_id'][sel_t1])
            labels.append(int(subject0 == subject1))
        return np.array(labels)


    def pool_feats_for_templates(features, template_set, metadata, sqrt=False,
                                 eps=1e-12):
        """Pool the features for each image contained in each template
        set. For each face belonging to a given template, the feautres
        produced by the CNN are averaged, normalised and then pooled.

        Args:
           features (pytorch.Tensor): NxDx1x1 tensor containing the
               D-dimensional embeddings of each of the N listings in the IJBA
               metadata for the current split.
           template_set (ndarray): the unique template identifiers (each of
               which can correspond to several faces).
            metadata (dict): all metadata associated with the verification
              evaluation.
            sqrt (bool) [False]: whether to transform the features by a signed
              square root operation.
            eps (float) [1e-12]: a small constant used for numerical stability.

        Returns:
            (pytorch.Tensor) the pooled features.
        """
        print('pooling templates...')
        pooled_features = []
        for tid in tqdm.tqdm(template_set):
            sel = np.where(metadata['template_id'] == tid)[0]
            feat = features[sel,:].unsqueeze(0).mean(1)
            if sqrt:
                feat = torch.mul(torch.sign(feat),
                                 torch.sqrt(torch.abs(feat) + eps))
            normalized = F.normalize(feat, p=2, dim=1)
            pooled_features.append(normalized)
        return torch.cat(pooled_features, dim=0)

    def compute_pair_similarities(template_set, pooled_features, pairs,
                                  cosine=False):
        """Compute the similarity scores between the features associated
        with each template set pairing. If Euclidean distances are used, the
        similarity is the negative distance between features.

        Args:
            template_set (ndarray): the unique template identifiers (each of
                which can correspond to several faces).
            pooled_features (pyptorch.Tensor): the averaged (and normalised)
               feature associated with each template.
            pairs (ndarray): Nx2 array containing the indices of each test
               pair, where N is the total number of pairs.
            cosine (bool) [False]: whether to compute cosine distance, rather
               than Euclidean distance between features (note that if the
               features are L2

        Returns:
            (ndarray): 1xN list of similarity scores (i.e negative distance)
        """
        print('computing pair distances...')
        scores = []
        for pair in tqdm.tqdm(pairs):
            sel_t0 = np.where(template_set == pair[0])[0]
            sel_t1 = np.where(template_set == pair[1])[0]
            f0 = torch.squeeze(pooled_features[sel_t0,:,:,:])
            f1 = torch.squeeze(pooled_features[sel_t1,:,:,:])
            if cosine:
                feat_sim = torch.dot(f0, f1)
            else:
                feat_sim = -(f0 - f1).norm(p=2, dim=0)
            scores.append(feat_sim.numpy()) # score: negative of L2-distance
        return np.array(scores)

    labels = compute_pair_labels(pairs, metadata)
    template_set = np.unique(metadata['template_id'])
    pooled_features = pool_feats_for_templates(features, template_set,
                                               metadata, sqrt=args.sqrt)
    scores = compute_pair_similarities(template_set, pooled_features, pairs,
                                       args.cosine)

    # Metrics: TAR (tpr) at FAR (fpr)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores)
    fpr_levels = [0.0001, 0.001, 0.01, 0.1]
    f_interp = interpolate.interp1d(fpr, tpr)
    tpr_at_fpr = [f_interp(x) for x in fpr_levels] # use linear interpolation

    for (far, tar) in zip(fpr_levels, tpr_at_fpr):
        print('TAR @ FAR={:0.4f} : {:0.4f}'.format(far, float(tar)))

    res = {}
    res['TAR'] = tpr_at_fpr
    res['FAR'] = fpr_levels
    res_path = osp.join(cache_dir, 'result-1-1-fold-%d.yaml'.format(fold_id))
    with open(res_path, 'w') as f:
        yaml.dump(res, f, default_flow_style=False)
    roc_curve_data = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds,
                      'tpr_at_fpr': tpr_at_fpr}
    roc_path = osp.join(cache_dir, 'roc-1-1-fold-{}.mat'.format(fold_id))
    sio.savemat(roc_path, roc_curve_data)

if __name__ == '__main__':
    main()
