## called from PYTHONSTARTUP environment variable
#import cv2
#import numpy as np
#import bobo.show
#from bobo.show import *
import matplotlib
matplotlib.use('Agg')
import numpy as np

from collections import OrderedDict
from sklearn import metrics
import matplotlib.pyplot
#from matplotlib.pyplot import plot
#import bobo.util
#from bobo.util import *
from scipy import interpolate
from os.path import join as pjoin
from scipy import io as sio

#import janus.metrics
import sys, os

# currently this is the only way to avoid segfaulting, need a better fix
opencv_dir = '/users/jdt/local/opencv/opencv-2.4.13.3/release/lib'
cwd = os.getcwd()
os.chdir(opencv_dir)
import cv2
os.chdir(cwd)

sys.path.insert(0, '/scratch/shared/nfs1/lishen/janus/bobo')
sys.path.insert(0, '/scratch/shared/nfs1/lishen/janus/python')
from janus.metrics_by_li import ijba11_multi

score_root = '/scratch/local/ssd/albanie/datasets/ijba/score_11'
#model_name = ['vgg2_atten_ft']
#curve_name = ['vgg2_atten_ft']
model_name = ['vgg2_senet50_ft-dag_mcn_ft']
curve_name = ['vgg2_senet50_ft-dag_mcn_ft']

#score_root = '/scratch/shared/nfs1/lishen/janus/ijba/verification/eval_result'
#model_name = ['vgg2_atten_scratch']
#curve_name = ['vgg2_atten_scratch']

#score_root = '/data1/janus_CS2/score_11/'
#model_name = ['vgg1_resnet50_',
#              'msra_',
#              'vgg2_resnet50_scratch_',
#              'vgg2_resnet50_',
#              'vgg2_atten_scratch_',
#              'vgg2_atten_']

#model_name = ['vgg2_atten_scratch_']
#model_name = ['vgg2_atten_']
#curve_name = ['VF_ResNet',
#              'MS1M_ResNet',
#              'VF2_ResNet',
#              'VF2_ft_ResNet',
#              'VF2_SENet',
#              'VF2_ft_SENet']

Ym = []
Yhatm = []
prependToLegendm = []
num_splits = 10


for k in range(0, len(curve_name)):
    Y = []
    Yhat = []
    model = model_name[k]

    fpr_levels = [0.0001, 0.001, 0.01, 0.1]
    tprs = OrderedDict()
    for fpr in fpr_levels:
        tprs[fpr] = []

    for ii in range(1, num_splits + 1): # split names are 1-indexed
        print(ii)
        #pdb.set_trace()
        src_path = pjoin(score_root, '{}_cos_avg_split{}.mat'.format(model, ii))
        #print('{}/{} loading: {}'.format(ii, num_splits, src_path))
        score = sio.loadmat(src_path)
        Y.append(score['Y'])
        Yhat.append(score['Yhat'])

        labels = score['Y']
        scores = score['Yhat']
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
        f_interp = interpolate.interp1d(fpr, tpr)
        tpr_at_fpr = [f_interp(x) for x in fpr_levels]

        for (far, tar) in zip(fpr_levels, tpr_at_fpr):
            #print('TAR @ FAR={:0.4f} : {:0.4f}'.format(far, float(tar)))
            tprs[far].append(tar)
    Ym.append(Y)
    Yhatm.append(Yhat)

    for far, tar_vals in tprs.items():
        mean_tar = np.array(tar_vals).mean()
        std_tar = np.array(tar_vals).std()
        print('TAR @ FAR={:0.4f} : {:0.4f} : {:0.4f}'.format(far, mean_tar, std_tar))

  #import ipdb ; ipdb.set_trace()

    prependToLegendm.append(curve_name[k])
    result = ijba11_multi(Ym=Ym, Yhm=Yhatm, prependToLegendm=prependToLegendm,
                         detLegendSwap=False, splitmean=True, hold=True,
                         mark=False, verbose=True)
