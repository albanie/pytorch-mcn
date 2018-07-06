function compute_matlab_feats(varargin)
%COMPUTE_MATLAB_FEATS - compute IJBA features with MatConvNet

	opts.useGpu = 1 ;
	opts.limit = inf ;
	opts.batchSize = 256 ;
	opts.featName = 'pool5_7x7_s1' ;
	%opts.modelName = 'resnet50_ft-dag' ;
	opts.modelName = 'senet50_ft-dag' ;
	opts.imDir = '/scratch/shared/nfs1/lishen/janus/ijba/verification/crop_verify' ;
	opts.modelDir = '/scratch/shared/nfs1/albanie/models/matconvnet/vggface2_models' ;
	opts.outputDir = '/scratch/shared/nfs1/albanie/pt/pytorch-mcn/ijba-feats-matlab' ;
	opts.imgListFile = '/scratch/shared/nfs1/lishen/janus/ijba/verification/verify_img.txt' ;
  opts = vl_argparse(opts, varargin) ;

	modelPath = fullfile(opts.modelDir, sprintf('%s.mat', opts.modelName)) ;
	destPath = fullfile(opts.outputDir, sprintf('%s-feats.mat', opts.modelName)) ;

  fprintf('loading %s from disk ...', opts.modelName) ; tic ;
  net = load(modelPath) ;
  dag = dagnn.DagNN.loadobj(net) ;
  fprintf('done in %g(s)\n', toc) ;

	imgList = importdata(opts.imgListFile) ;
	vIdx = dag.getVarIndex(opts.featName) ;
	dag.vars(vIdx).precious = true ;
  dag.mode = 'test' ;

	if opts.useGpu
		if strcmp(dag.device, 'cpu'), dag.move('gpu') ; end
	end

	inVars = dag.getInputs() ;
	assert(numel(inVars) == 1, 'too many inputs') ;
	numIms = min(numel(imgList), opts.limit) ;
	feats = zeros(numIms, 2048, 'single') ;

	for ii = 1:opts.batchSize:numIms
		tic ;
		batchStart = ii ;
		batchEnd = min(batchStart+opts.batchSize-1, numIms) ;
		batch = batchStart:batchEnd ;
		imPaths = fullfile(opts.imDir, imgList(batch)) ;
		data = getImageBatch(imPaths, dag) ;
		dag.eval({inVars{1}, data{1}}) ;
		feat = dag.vars(vIdx).value ;
		feats(batch,:) = squeeze(gather(feat))' ;
		rate = numel(batch) / toc ;
		etaStr = zs_eta(rate, ii, numIms) ;
		fprintf('processed image %d/%d at (%.3f Hz) (%.3f%% complete) (eta:%s)\n', ...
					 ii, numIms, rate, 100 * ii/numIms, etaStr) ;
	end

	if opts.useGpu, dag.move('cpu') ; end
	fprintf('saving feats to %s ...', destPath) ; tic ;
	save(destPath, 'feats') ;
	fprintf('done in %g(s)\n', toc) ;
end

% --------------------------------------------------------
function data = getImageBatch(imagePaths, dag)
% --------------------------------------------------------
  numThreads = 10 ;
  opts.prefetch = false ; % can optimise this
  avgIm = dag.meta.normalization.averageImage ;
  imageSize = dag.meta.normalization.imageSize(1:2) ;

  args{1} = {imagePaths, ...
             'NumThreads', numThreads, ...
             'Gpu', ...
             'Pack', ...
             'SubtractAverage', avgIm, ...
             'Interpolation', 'bilinear', ... % use bilinear to reproduce trainig resize
             'Resize', imageSize, ...
             'CropLocation', 'center'} ; % centre crop for testing
  args = horzcat(args{:}) ;

  if opts.prefetch
    vl_imreadjpeg(args{:}, 'prefetch') ;
    data = [] ;
  else
    data = vl_imreadjpeg(args{:}) ;
  end
end
