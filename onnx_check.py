"""
For now, onnx doesn't have sufficient support to cover model
debugging - it may be worthwhile returning to in future
"""
from torch.autograd import Variable
import torch.onnx
from torchvision import models

dummy_input = Variable(torch.randn(10, 3, 224, 224))
model = models.squeezenet.squeezenet1_0(pretrained=True)
# model = models.squeeznet.squeeznet1_0(pretrained=True)
torch.onnx.export(model, dummy_input, "weights/squeeznet1_0.proto", verbose=True)
