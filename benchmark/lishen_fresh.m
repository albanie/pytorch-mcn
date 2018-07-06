%load information
%root_dir = '/data1/janus_CS2/';
%model_dir = '/data1/janus_eval_model/ft/'
% model_dir = '/data2/vggface1/model/'

  opts.feat_nl = true ;
  root_dir = '/scratch/local/ssd/albanie/datasets/ijba' ;
  %featPath =  '/scratch/shared/nfs1/albanie/pt/pytorch-mcn/ijba-feats-matlab/resnet50_ft-dag-feats.mat' ;

  % model_dir = '/data2/vggface1/model/'
  protocol_dir = fullfile(root_dir, 'protocol/IJB-A_11') ;
  save_root_dir = fullfile(root_dir, 'score_11') ;
  verify_database = readtable(fullfile(root_dir, 'verification/verify.csv')) ;
  if ~exist(opts.save_root_dir, 'dir'), mkdir(opts.save_root_dir) ; end

train_svm = false;
feat_dim = 2048;
feat_nl = true;
use_face_score = false; score_ = 0;
weighted_face_score = false; threshold = 0.7;
sample_num = 51090;
use_size_filter = false; size_ = 30; % 30 is better than 50
save_name =  'vgg2_atten_li';
%feat_name = fullfile(save_name, 'cs2/CS2_verify.bin');
feat_name = '/scratch/shared/nfs1/lishen/janus/ijba/example/CS2_verify.bin' ;
% feat_name = fullfile('vggface1_verify.mat');
disp(feat_name);
fprintf('read feature\n');
%f_ID = fopen(fullfile(model_dir, feat_name));
f_ID = fopen(feat_name);
features = single(fread(f_ID, [feat_dim sample_num], 'single'));
% features = importdata(feat_name);

if feat_nl
    fprintf('feature normalization\n');
    features = bsxfun(@times, features, 1./max(sqrt(sum(features.^2)), 1e-10));
end
if use_size_filter
    width = verify_database.WIDTH;
    height = verify_database.HEIGHT;
    % remove the too small faces
    is_remove = find(min(width, height) < size_);
end
tid = verify_database.TID;
%faceid = verify_database.faceid;
sid = verify_database.SID;
mid = verify_database.MID;
% generate feature vectors by mid
unique_mid = unique(mid, 'stable');
if use_face_score % directly remove unface (two many empty ones)
   face_score = importdata([root_dir 'face_score.txt']);
   is_nonface = find(face_score == 0);
end
feat_vect = zeros(feat_dim, numel(unique_mid));
% generate mapping between utid and usid
uTid = unique(tid, 'stable');
num_tid = numel(uTid);
map_tid_sid = zeros(num_tid,2);
for i = 1:num_tid
    map_tid_sid(i,1) = uTid(i);
    ind = find(tid == uTid(i));
    tt_sid = unique(sid(ind));
    assert(numel(tt_sid) == 1);
    map_tid_sid(i,2) = tt_sid;
end
fprintf('computer per-mid features\n');
is_zero_feat = zeros(1, numel(unique_mid));
for i = 1:numel(unique_mid)
    ind = find(mid == unique_mid(i));
    if  weighted_face_score
        t_score = face_score(ind);
        tt_feat = features(:, ind);
        if threshold >  0
            t_score (t_score > threshold) = 0;
        end
%         if isempty(t_score > 0)
%             feat_vect(:, i) = mean(tt_feat, 2);
%         else
%             feat_vect(:, i) = mean(tt_feat.* repmat(t_score',size(features,1) ,1), 2);
%         end
        if ~isempty(t_score > 0)
            feat_vect(:, i) = mean(tt_feat.* repmat(t_score', size(features, 1), 1), 2);
        end
    elseif use_size_filter || use_face_score
        tt_ind = ind;
        if use_size_filter
            tt_ind = setdiff(tt_ind, is_remove);
        end
        if use_face_score
            tt_ind = setdiff(tt_ind, is_nonface);
        end
        if ~isempty(tt_ind)
            tt_feat = features(:, tt_ind);
            feat_vect(:, i) = mean(tt_feat, 2);
        else
            is_zero_feat(i) = 1;
            tt_feat = features(:, ind);
            feat_vect(:, i) = mean(tt_feat, 2);
        end
    else
        tt_feat = features(:, ind);
        feat_vect(:, i) = mean(tt_feat, 2);
    end
end
fprintf('compute per-template features\n');
 % generate single feat for each template
template_feat = zeros(feat_dim, num_tid);
for i = 1:num_tid
    % sample index
    tt_tid_ind = find(tid == uTid(i));
    tt_mid = unique(mid(tt_tid_ind),'stable');
%     if use_size_filter || use_face_score
%         tt_ind = ismember(unique_mid, tt_mid) & ~is_zero_feat;
%         if sum(tt_ind) > 0 % non-empty
%             pos_feat = feat_vect(:, tt_ind);
%             template_feat(:, i) = mean(pos_feat, 2);
%         end
%     else
    pos_feat = feat_vect(:, ismember(unique_mid,tt_mid));
    template_feat(:, i) = mean(pos_feat, 2);
%     end
end
template_feat = bsxfun(@times, template_feat, 1./max(sqrt(sum(template_feat.^2)), 1e-10));

%% evaluation for each split of 10
for i = 1:10
    % load pair list
    fprintf('read table %d\n', i);
    X = readtable(fullfile(protocol_dir, ['split',num2str(i)], ['verify_comparisons_', num2str(i), '.csv']), 'ReadVariableNames',false);
    p1 = X{:, 'Var1'};
    p2 = X{:, 'Var2'};
    num_pairs = numel(p1);
    true_label = zeros(1, num_pairs) -1;
    % tid and mid not including relation
%     if train_svm
%         score_avg_svm = zeros(num_pairs, 1);
%         run(fullfile(root_dir, 'CS4_svm/vlfeat-0.9.20/toolbox/vl_setup'));
%         % train svm for each template
%         model = cell(num_tid,1);
%         neg_feat = double(importdata(fullfile(root_dir, 'matlab/str_resnet50_v1.mat'))');
%         %svm configuration
%         lambda = 1/size(neg_feat,2); maxIter = 10^6; C_ = 0.2;
%         parfor i = 1:num_tid
%             fprintf('Train %d svm \n', i);
%             pos_feat = template_feat(:,i);
%             tm_labels = -ones(1 + size(neg_feat,2),1);
%             tm_labels(1) = 1;
%             C_pos = C_ * numel(tm_labels)/ (2 * size(pos_feat,2));
%             C_neg = C_ * numel(tm_labels)/ (2 * size(neg_feat,2));
%             tm_weights = ones(numel(tm_labels),1);
%             tm_weights(1) = C_pos;
%             tm_weights(2: end) = C_neg;
%              [w,b,~] = vl_svmtrain([pos_feat, neg_feat], tm_labels, lambda, ...
%                  'maxNumIterations',maxIter,'loss','hinge2','biaslearningrate',0,...
%                  'biasmultiplier', 0, 'solver','sgd','weights',tm_weights);
%             model{i} = w';
%             pred_ = w'*[pos_feat, neg_feat]+ b;
%             sum(tm_labels(pred_> 0))
%             size(pos_feat,2)
%         end
%     %%    generate groundtruth and compute scores
%         parfor i = 1: num_pairs
%             i
%             p1_tid_ind = find(uTid == p1(i));
%             p2_tid_ind = find(uTid == p2(i));
%             pair_1_feat = double(template_feat(:, p1_tid_ind));
%             pair_2_feat = double(template_feat(:, p2_tid_ind));
%             pair_1_sid = map_tid_sid(p1_tid_ind, 2);
%             pair_2_sid = map_tid_sid(p2_tid_ind, 2);
%             true_label(i) = (pair_1_sid == pair_2_sid);
%             p1_svm = model{p1_tid_ind};
%             pair_matrix1 = p1_svm * pair_2_feat;
%             p2_svm = model{p2_tid_ind};
%             pair_matrix2 = p2_svm * pair_1_feat;
%             score_avg_svm(i) = 0.5* (pair_matrix1 + pair_matrix2);
%         end
%         Y = true_label';
%         Yhat = score_avg_svm;
%         save(fullfile(save_root_dir, [save_name, '_avg_feat_svm.mat']), 'Y', 'Yhat');
%
%     else
        %directly compute by similarity
    %     score_avg_l2 = zeros(1, num_pairs);
        score_avg_cos = zeros(1, num_pairs);
    %     %generate groundtruth and compute scores
    %     score_matrix = zeros(num_tid, num_tid);
    %     for i = 1:num_tid
    %         i
    %         score_matrix(i,:) = template_feat(:,i)'* template_feat;
    %     end
        parfor jj = 1: num_pairs
            jj
            p1_tid_ind = find(uTid == p1(jj));
            p2_tid_ind = find(uTid == p2(jj));
            pair_1_feat = double(template_feat(:, p1_tid_ind)');
            pair_2_feat = double(template_feat(:, p2_tid_ind)');
            pair_1_sid = map_tid_sid(p1_tid_ind, 2);
            pair_2_sid = map_tid_sid(p2_tid_ind, 2);
            true_label(jj) = (pair_1_sid == pair_2_sid);

    %         pair_matrix_l2 = pdist2(pair_1_feat,pair_2_feat,'euclidean');
            pair_matrix_cos = pair_1_feat * pair_2_feat';
    %         score_avg_l2(i) = pair_matrix_l2;
            score_avg_cos(jj) = pair_matrix_cos;
    %         score_avg_cos(i) = score_matrix(p1_tid_ind, p2_tid_ind);
        end
        Y = true_label';
    %     Yhat = max(score_avg_l2) -  score_avg_l2';
    %     save([root_dir,'eval_result/', save_name, '_avg_l2_avg_feat.mat'], 'Y', 'Yhat');
        Yhat = score_avg_cos';
        destPath = fullfile(save_root_dir, ...
                        [save_name, '_ft_cos_avg_split', num2str(i), '.mat']) ;
        save(destPath, 'Y', 'Yhat');
        keyboard
%     end
end
