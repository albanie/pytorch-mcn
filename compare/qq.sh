mcn_import_dir="~/data/models/matconvnet"
output_dir="~/data/models/pytorch/mcn_imports"
feat_dir="~/data/pt/pytorch-mcn/feats"
ipython compare_models.py -- $mcn_import_dir/inception_v3-pt-mcn.mat $output_dir $feat_dir
