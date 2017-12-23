function startup
%STARTUP - setup local paths for matconvnet
%
% Licensed under The MIT License [see LICENSE.md for details]
% Copyright (C) 2017 Samuel Albanie

  % modify the paths below to match your installation
  base = '~/coding/libs/matconvnets/contrib-matconvnet' ;
  run([base '/matlab/vl_setupnn']) ;
  run([base '/contrib/mcnExtraLayers/setup_mcnExtraLayers']) ;
  run([base '/contrib/autonn/setup_autonn']) ;
