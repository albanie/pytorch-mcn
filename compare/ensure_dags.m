function ensure_dags
%ENSURE_DAGS - ensure that pretrained models are in dagnn format
%   ENSURE_DAGS checks (and if necessary modifies) a selection of
%   matconvnet models and ensures that they are stored in the DagNN
%   wrapper format.
%
% Licensed under The MIT License [see LICENSE.md for details]
% Copyright (C) 2017 Samuel Albanie

  pretrained = {...
    'imagenet-matconvnet-vgg-f', ...
    'imagenet-matconvnet-vgg-m', ...
    'imagenet-matconvnet-vgg-s' ...
    'imagenet-matconvnet-vgg-verydeep-16' ...
  } ;

  modelDir = '~/data/models/matconvnet/' ;
  for ii = 1:numel(pretrained)
    modelName = pretrained{ii} ;
    fprintf('converting %s to dagnn... \n', modelName) ;
    srcPath = fullfile(modelDir, sprintf('%s.mat', modelName)) ;
    destPath = fullfile(modelDir, sprintf('%s-dag.mat', modelName)) ;
    if exist(destPath, 'file')
      fprintf('%s exists, skipping\n', destPath) ;
      continue
    end
    net = load(srcPath) ;
    dag = dagnn.DagNN.fromSimpleNN(net) ;
    net = dag.saveobj() ; %#ok
    save(destPath, '-struct', 'net') ;
  end
