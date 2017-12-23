function generateFeats(modelPath, destPath, varargin)
% GENERATEFEATS - generate intermediate network features for comparison
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
%
% Licensed under The MIT License [see LICENSE.md for details]
% Copyright (C) 2017 Samuel Albanie

  opts.gpus = [] ;
  opts.sampleImPath = 'BdM.jpg' ;
  opts = vl_argparse(opts, varargin) ;

  net = load(modelPath) ;
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
