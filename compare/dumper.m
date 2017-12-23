function dumper(varargin)
%DUMPER - dump intermediate network features
%   DUMPER computes and stores intermediate network features for a given
%   set of matconvnet networks.
%
% Licensed under The MIT License [see LICENSE.md for details]
% Copyright (C) 2017 Samuel Albanie

  modelDir = '~/data/models/matconvnet' ;
  featDir = '~/data/pt/pytorch-mcn/feats' ;

  % Declare list of matconvnet models to be checked
  modelNames = {'squeezenet1_0-pt-mcn'} ;

  for ii = 1:numel(modelNames)
    modelPath = fullfile(modelDir, sprintf('%s.mat', modelNames{ii})) ;
    featPath = fullfile(featDir, sprintf('%s-feats.mat', modelNames{ii})) ;
    generateFeats(modelPath, featPath) ;
  end
