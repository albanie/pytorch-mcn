% sanity check python results with Li Shen's script

%load information
%root_dir = '/data1/janus_CS2/';
%model_dir = '/data1/janus_eval_model/ft/'

  opts.feat_nl = true ;
  %modelName = 'senet50_ft-dag' ;
  modelName = 'resnet50_ft-dag' ;
  root_dir = '/scratch/local/ssd/albanie/datasets/ijba' ;
  featDir = '/scratch/shared/nfs1/albanie/pt/pytorch-mcn/ijba-feats-matlab' ;
  featPath = fullfile(featDir, sprintf('%s-feats.mat', modelName)) ;

  % model_dir = '/data2/vggface1/model/'
  protocol_dir = fullfile(root_dir, 'protocol/IJB-A_11') ;
  opts.save_root_dir = fullfile(root_dir, 'score_11') ;
  verify_database = readtable(fullfile(root_dir, 'verification/verify.csv')) ;
  if ~exist(opts.save_root_dir, 'dir'), mkdir(opts.save_root_dir) ; end

  %configuration
  score_ = 0 ;
  size_ = 30 ; % 30 is better than 50
  opts.eps = 1e-10 ;
  threshold = 0.7 ;
  feat_dim = 2048 ;
  train_svm = false ;
  sample_num = 51090 ;
  opts.use_face_score = false ;
  opts.framework = 'mcn' ;
  opts.weighted_face_score = false ;
  opts.use_size_filter = false ;

  opts.save_name =  sprintf('vgg2_%s_%s', modelName, opts.framework) ;

  switch opts.framework
  case 'mcn'
    fprintf('loading features...') ;  tic ;
    tmp = load(featPath, 'feats') ;
    features = tmp.feats' ; % this script expects the feature dim first
    fprintf('done in %g(s) \n', toc) ;
  case 'caffe'
    feat_name = '/scratch/shared/nfs1/lishen/janus/ijba/example/CS2_verify.bin' ;
    f_ID = fopen(feat_name) ;
    features = single(fread(f_ID, [feat_dim sample_num], 'single')) ;
    fclose(f_ID) ;
  otherwise, error('% not recognised\n', opts.framework) ;
  end

  if opts.feat_nl
    fprintf('applying feature normalization\n') ;
    features = l2norm(features, opts.eps) ;
  end

  if opts.use_size_filter
    width = verify_database.WIDTH ;
    height = verify_database.HEIGHT ;
    is_remove = find(min(width, height) < size_) ; % remove very small faces
  end

  %faceid = verify_database.faceid ;
  tid = verify_database.TID ;
  sid = verify_database.SID ;
  mid = verify_database.MID ;

  % generate feature vectors by mid
  unique_mid = unique(mid, 'stable') ;

  if opts.use_face_score % directly remove nonfaces (too many empty ones)
     face_score = importdata([root_dir 'face_score.txt']) ;
     is_nonface = find(face_score == 0) ;
  end

  feat_vect = zeros(feat_dim, numel(unique_mid)) ;

  % -------------------------------
  % Murky
  % -------------------------------

  % generate mapping between unique template identifiers and unique sighting
  % ids (i.e. tids and usids)
  uTid = unique(tid, 'stable') ; % unique template identifiers
  num_tid = numel(uTid) ;
  map_tid_sid = zeros(num_tid, 2) ;

  % generate pairs
  for ii = 1:num_tid
    map_tid_sid(ii, 1) = uTid(ii) ;
    ind = find(tid == uTid(ii)) ;
    tt_sid = unique(sid(ind), 'stable') ;
    assert(numel(tt_sid) == 1) ;
    map_tid_sid(ii,2) = tt_sid ;
  end

  fprintf('computer per-mid features\n') ;
  is_zero_feat = zeros(1, numel(unique_mid)) ;

  % average features across video frames, so that they don't carry more weight
  % than individual images
  for ii = 1:numel(unique_mid)
    ind = find(mid == unique_mid(ii)) ;
    if opts.weighted_face_score
      t_score = face_score(ind) ;
      tt_feat = features(:, ind) ;
      if threshold >  0, t_score(t_score > threshold) = 0 ; end
        if isempty(t_score > 0)
          feat_vect(:, ii) = mean(tt_feat, 2) ;
        else
          feat_vect(:, ii) = mean(tt_feat.* repmat(t_score',size(features,1) ,1), 2) ;
        end
      if ~isempty(t_score > 0)
        feat_vect(:, ii) = mean(tt_feat.* repmat(t_score', size(features, 1), 1), 2) ;
      end
    elseif opts.use_size_filter || opts.use_face_score
      tt_ind = ind ;
      if opts.use_size_filter
        tt_ind = setdiff(tt_ind, is_remove) ;
      end
      if opts.use_face_score
        tt_ind = setdiff(tt_ind, is_nonface) ;
      end
      if ~isempty(tt_ind)
        tt_feat = features(:, tt_ind) ;
        feat_vect(:, ii) = mean(tt_feat, 2) ;
      else
        is_zero_feat(ii) = 1 ;
        tt_feat = features(:, ind) ;
        feat_vect(:, ii) = mean(tt_feat, 2) ;
      end
    else
      tt_feat = features(:, ind) ;
      feat_vect(:, ii) = mean(tt_feat, 2) ;
    end
  end

  fprintf('compute per-template features\n') ;
   % generate single feat for each template
  template_feat = zeros(feat_dim, num_tid) ;

  % loop over each unique template identifier
  for ii = 1:num_tid
      % sample index
    tt_tid_ind = find(tid == uTid(ii)) ;
    tt_mid = unique(mid(tt_tid_ind), 'stable') ; % find matching media ids
  %     if use_size_filter || use_face_score
  %         tt_ind = ismember(unique_mid, tt_mid) & ~is_zero_feat ;
  %         if sum(tt_ind) > 0 % non-empty
  %             pos_feat = feat_vect(:, tt_ind) ;
  %             template_feat(:, ii) = mean(pos_feat, 2) ;
  %         end
  %     else
    keep = ismember(unique_mid, tt_mid) ;
    pos_feat = feat_vect(:, keep) ;
    template_feat(:, ii) = mean(pos_feat, 2) ;
  %     end
  end

  template_feat = l2norm(template_feat, opts.eps) ;

  % evaluation for each split of 10
  numSplits = 10 ;
  for ii = 1:numSplits
    fprintf('read table %d\n', ii) ; % load pair list
    csvPath = fullfile(protocol_dir, sprintf('split%d', ii), ...
                           sprintf('verify_comparisons_%d.csv', ii)) ;
    X = readtable(csvPath, 'ReadVariableNames', false) ;
    p1 = X{:, 'Var1'} ; p2 = X{:, 'Var2'} ;
    numPairs = numel(p1) ;
    true_label = zeros(1, numPairs) - 1 ;
    score_avg_cos = zeros(1, numPairs) ;

    parfor jj = 1: numPairs
      fprintf('(%d/%d) parfor processing %d/%d\n', ...
                                ii, numSplits, jj, numPairs) ;
      p1_tid_ind = find(uTid == p1(jj)) ;
      p2_tid_ind = find(uTid == p2(jj)) ;
      pair_1_feat = double(template_feat(:, p1_tid_ind)') ;
      pair_2_feat = double(template_feat(:, p2_tid_ind)') ;
      pair_1_sid = map_tid_sid(p1_tid_ind, 2) ;
      pair_2_sid = map_tid_sid(p2_tid_ind, 2) ;
      true_label(jj) = (pair_1_sid == pair_2_sid) ;
      pair_matrix_l2 = pdist2(pair_1_feat, pair_2_feat, 'euclidean') ;
      pair_matrix_cos = pair_1_feat * pair_2_feat' ;
      score_avg_cos(jj) = pair_matrix_cos ;
    end

    tmp.Y = true_label' ;
    tmp.Yhat = score_avg_cos' ;
    cosDestPath = fullfile(opts.save_root_dir, ...
                      sprintf('%s_ft_cos_avg_split%d.mat', opts.save_name, ii))  ;
    save(cosDestPath, '-struct', 'tmp') ;
  end

% -------------------------------------------------------------------
function y = l2norm(x, eps)
% -------------------------------------------------------------------
  y = bsxfun(@times, x, 1./max(sqrt(sum(x.^2)), eps)) ;
end
