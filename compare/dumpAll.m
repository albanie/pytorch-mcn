%modelNames = {...
%'alexnet-pt-mcn' ...
%'vgg11-pt-mcn' ...
%'vgg13-pt-mcn' ...
%'vgg16-pt-mcn' ...
%'vgg19-pt-mcn' ...
%} ;
%'resnet18-pt-mcn', ...
%'resnet34-pt-mcn', ...
%'resnet50-pt-mcn', ...
%'resnet101-pt-mcn', ...
%'resnet152-pt-mcn' ...
%'inception_v3-pt-mcn'
%'densenet121-pt-mcn' ...
%'densenet161-pt-mcn' ...
%'densenet169-pt-mcn' ...
%'densenet201-pt-mcn' ...
modelNames = {...
'imagenet-matconvnet-alex'
} ;

featureDumper('modelNames', modelNames) ;
