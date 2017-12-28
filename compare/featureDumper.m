function featureDumper(varargin)
%FEATUREDUMPER - dump intermediate network features
%   FEATUREDUMPER computes and stores intermediate network features for a
%   given set of matconvnet networks. Note that the name of the model is
%   sanitized to work with Python, by replacing hyphens with underscores.
%
%   FEATUREDUMPER(..., 'option', value, ...) accepts the following
%   options:
%
%  `modelDir` :: `~/data/models/matconvnet`
%   Path to the directory containing matconvnet models.
%
%  `featDir` :: '~/data/pt/pytorch-mcn/feats'
%   Path to directory where intermediate features will be stored.
%
%  `modelNames` :: {'squeezenet1_0-pt-mcn'}
%   A cell array of target model names to be processed.
%
% Licensed under The MIT License [see LICENSE.md for details]
% Copyright (C) 2017 Samuel Albanie

  opts.modelDir = '~/data/models/matconvnet' ;
  opts.featDir = '~/data/pt/pytorch-mcn/feats' ;
  opts.modelNames = {'squeezenet1_0-pt-mcn'} ;
  opts = vl_argparse(opts, varargin) ;

  for ii = 1:numel(opts.modelNames)
    modelName = opts.modelNames{ii} ;
    fprintf('dumping features for %s\n', modelName) ;
    modelPath = fullfile(opts.modelDir, sprintf('%s.mat', modelName)) ;
    featFile = sprintf('%s-feats.mat', strrep(modelName, '-', '_')) ;
    featPath = fullfile(opts.featDir, featFile) ;
    generateFeats(modelPath, featPath) ;
  end

% ------------------------------------------------------------------------
function generateFeats(modelPath, destPath, varargin)
% ------------------------------------------------------------------------
%GENERATEFEATS - generate intermediate network features for comparison
%   GENERATEFEATS(MODELPATH, DESTPATH) - loads the matconvnet model found
%   at MODELPATH, applies it to a sample image and stores the resulting
%   features at DESTPATH.
%
%   GENERATEFEATS(.., 'name', value, ...) accepts the following options:
%
%   `gpus` :: []
%    Gpu device indices for performing a forward pass over the network.
%
%   `sampleImPath`: 'BdM.jpg'
%    Path to a sample RGB image for processing.
%
%    NOTE:
%    The default sample image original can be found at:
%    url = ${base}#/media/File:Modigliani_-_Busto_de_mulher.jpg
%    where ${base} = https://en.wikipedia.org/wiki/Amedeo_Modigliani

  opts.gpus = [] ;
  opts.sampleImPath = 'BdM.jpg' ;
  opts = vl_argparse(opts, varargin) ;

  net = load(modelPath) ;
  net = fixBackwardsCompatibility(net) ;
  dag = dagnn.DagNN.loadobj(net) ;
  dag.conserveMemory = false ;

  im = single(imread(opts.sampleImPath)) ;
  im = imresize(im, dag.meta.normalization.imageSize(1:2)) ;

  if ~isempty(opts.gpus)
    gpuDevice(opts.gpus) ; im = gpuArray(im) ; dag.move('gpu') ;
  end

  dag.mode = 'test' ;
  dag.eval({dag.getInputs{1}, im}) ;
  featStore = struct() ;
  for ii = 1:numel(dag.vars)
    featStore.(dag.vars(ii).name) = gather(dag.vars(ii).value) ;
  end
  if ~exist(fileparts(destPath), 'dir'), mkdir(fileparts(destPath)) ; end
  save(destPath, '-struct', 'featStore') ;

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
