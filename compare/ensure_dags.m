function ensure_dags
%ENSURE_DAGS - ensure that pretrained models are in dagnn format
%   ENSURE_DAGS checks (and if necessary modifies) a selection of
%   matconvnet models and ensures that they are stored in the DagNN
%   wrapper format. It also ensures that the meta properties of the
%   model conform to a consistent interface (i.e. that the
%   average RGB training image is stored as a [1 1 3] dimensional array).
%
% Licensed under The MIT License [see LICENSE.md for details]
% Copyright (C) 2017 Samuel Albanie

  pretrained = {...
    'imagenet-vgg-f', ...
    'imagenet-vgg-m-1024', ...
    'imagenet-matconvnet-vgg-f', ...
    'imagenet-matconvnet-vgg-m', ...
    'imagenet-matconvnet-vgg-s', ...
    'imagenet-matconvnet-vgg-verydeep-16', ...
		'alexnet-face-bn', ...
		'alexnet-face-fer-bn', ...
		'vgg-m-face-bn', ...
		'vgg-vd-face-fer', ...
		'vgg-vd-face-sfew', ...
		'resnet50-face-bn', ...
		'resnet50-face-sfew' ...
  } ;
  pretrained = {...
		'vgg_face' ...
		'vgg-m-face-bn' ...
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
    if isfield(net, 'net'), net = net.net ; end
    if iscell(net.layers)
      net.layers(cellfun(@isempty, net.layers)) = [] ; % remove empty layers
      for jj = 1:numel(net.layers)
        name = net.layers{jj}.name ;
        if contains(name, ':')
          updated = strrep(name, ':', '_') ;
          fprintf('updating layer name %s to %s for model %s\n', ...
                             name, updated, modelName) ;
          net.layers{jj}.name = updated ;
        end
      end
      dag = dagnn.DagNN.fromSimpleNN(net) ;
    else
      dag = dagnn.DagNN.loadobj(net) ;
    end

    if numel(dag.meta.normalization.averageImage) > 3
      avgIm = dag.meta.normalization.averageImage ;
      avgIm = mean(mean(avgIm, 1), 2) ;
      assert(isequal(size(avgIm), [1 1 3]), 'unexpcted average image size')
      dag.meta.normalization.averageImage = avgIm ;
    end
    net = dag.saveobj() ; %#ok
    save(destPath, '-struct', 'net') ;
  end
