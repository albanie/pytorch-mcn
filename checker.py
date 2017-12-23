# from IPython import get_ipython
# ipython = get_ipython()
# ipython.magic('load_ext autoreload')
# ipython.magic('autoreload 2')

import torch
from torch.autograd import Variable
from torchvision import models
from models.squeezenet1_0 import squeezenet1_0

weights_path = 'weights/squeezenet1_0.pth'
# net = squeezenet1_0(weights_path=None)

net1 = squeezenet1_0(weights_path=weights_path)
net2 = models.squeezenet.squeezenet1_0(pretrained=True)

x = Variable(torch.randn((1,3,224,224)))

net1.eval()
net2.eval()

# y1 = net1(x)
y1 = net1.forward_debug(x)
y2 = net2(x)

params1 = list(net1.parameters())
params2 = list(net2.parameters())

for p1, p2 in zip(params1, params2):
    s1 = p1.size()
    s2 = p2.size()
    for x,y in zip(s1, s2):
        assert x == y
    eqs = torch.eq(p1,p2)
    if (torch.eq(p1, p2) == 0).sum() > 0:
        print('no match')
        import ipdb ; ipdb.set_trace()

# p1 = params1[0]
# p2 = params2[0]
