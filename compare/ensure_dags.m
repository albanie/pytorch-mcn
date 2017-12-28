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
    net = dag.saveobj() ;
    save(destPath, '-struct', 'net') ;
  end
