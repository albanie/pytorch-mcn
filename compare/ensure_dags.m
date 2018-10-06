function ensure_dags(varargin)
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
  pretrained = {...
		'vgg_face' ...
		'vgg-m-face-bn' ...
  } ;
  pretrained = {...
     'alexnet-face-fer-bn', ...
     'vgg-m-face-bn-fer', ...
     'vgg-vd-face-fer', ...
     'vgg-vd-face-sfew', ...
     'resnet50-face-sfew', ...
  } ;
  pretrained = {...
      'resnet50-ferplus', ...
      'senet50-ferplus', ...
  } ;
  opts.pretrained = {'resnet50_scratch-dag'} ;
  opts.modelDir = '~/data/models/matconvnet/' ;
  opts.destModelDir = '~/data/models/matconvnet/' ;
  opts.refresh = false ;
  opts = vl_argparse(opts, varargin) ;

  for ii = 1:numel(opts.pretrained)
    modelName = opts.pretrained{ii} ;
    fprintf('converting %s to dagnn... \n', modelName) ;
    srcPath = fullfile(opts.modelDir, sprintf('%s.mat', modelName)) ;
    destPath = fullfile(opts.destModelDir, sprintf('%s-dag.mat', modelName)) ;
		if exist(destPath, 'file') && ~opts.refresh
      fprintf('%s exists, skipping\n', destPath) ;
      continue
		end
    net = fixBackwardsCompatibility(load(srcPath)) ;
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

    if any(ismember(dag.getInputs(), 'input'))
      dag.renameVar('input', 'data') ;
    end

    if numel(dag.meta.normalization.averageImage) > 3
      avgIm = dag.meta.normalization.averageImage ;
      avgIm = mean(mean(avgIm, 1), 2) ;
      assert(isequal(size(avgIm), [1 1 3]), 'unexpcted average image size')
      dag.meta.normalization.averageImage = avgIm ;
    end
    dag.rebuild() ;
    dag.layers = dag.layers(dag.getLayerExecutionOrder()) ; % enforce
    net = dag.saveobj() ; %#ok
    save(destPath, '-struct', 'net') ;
  end

% ----------------------------------------------------------
function net = fixBackwardsCompatibility(net)
% ----------------------------------------------------------
%FIXBACKWARDSCOMPATIBILITY - remove unsupported attributes
%  NET = FIXBACKWARDSCOMPATIBILITY(NET) enables backwards
%  compatibility by remvoing attributes that are no longer
%  supported.

  removables = {'exBackprop'} ;
  for ii = 1:numel(net.layers)
    for jj = 1:numel(removables)
      fieldname = removables{jj} ;
      if isfield(net.layers(ii).block,fieldname)
        net.layers(ii).block = rmfield(net.layers(ii).block, fieldname) ;
      end
    end
  end
