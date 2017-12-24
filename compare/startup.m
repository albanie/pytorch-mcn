function startup
%STARTUP - setup local paths for matconvnet
%  STARTUP sets up the environment for numerically verifying models imported
%  from MatConvNet.  It expects the following to be installed:
%
%    1. MatConvNet (https://github.com/vlfeat/matconvnet)
%    2. Autonn (https://github.com/vlfeat/autonn)
%    3. mcnExtraLayers (https://github.com/albanie/mcnExtraLayers)
%
% Licensed under The MIT License [see LICENSE.md for details]
% Copyright (C) 2017 Samuel Albanie

  % modify the paths below to match your installation
  base = '~/coding/libs/matconvnets/contrib-matconvnet' ;
  run([base '/matlab/vl_setupnn']) ;
  run([base '/contrib/mcnExtraLayers/setup_mcnExtraLayers']) ;
  run([base '/contrib/autonn/setup_autonn']) ;
