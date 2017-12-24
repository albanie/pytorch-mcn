# importer.sh
# Import matconvnet models into pytorch
#
# --------------------------------------------------------
# pytorch-mcn
# Licensed under The MIT License [see LICENSE.md for details]
# Copyright (C) 2017 Samuel Albanie
# --------------------------------------------------------

# set paths/options
refresh_models=true
debug_mode=true
use_ipython=true
mcn_import_dir="~/data/models/matconvnet"
output_dir="~/data/models/pytorch/mcn_imports"

# verification options
verify_model=true
feat_dir="~/data/pt/pytorch-mcn/feats"

# Declare list of models to be imported (uncomment selection to run)
#declare -a model_list=("squeezenet1_0-pt-mcn")
declare -a model_list=("squeezenet1_1-pt-mcn")

pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null


function convert_model()
{
    mcn_model_path=$1
	echo "Exporting MatConvNet model to PyTorch (may take some time)..."
    if [ $use_ipython = true ] ; then
        converter="ipython $SCRIPTPATH/python/importer.py --"
    else
        converter="python $SCRIPTPATH/python/importer.py"
    fi
    if [ $refresh_models = true ] ; then
        opts="--refresh"
    fi
    if [ $debug_mode = true ] ; then
        opts="$opts --debug_mode"
    fi
    $converter $mcn_model_path $output_dir $opts

    if [ $verify_model = true ] ; then
        if [ $use_ipython = true ] ; then
            verifier="ipython $SCRIPTPATH/compare/compare_models.py --"
        else
            verifier="python $SCRIPTPATH/compare/compare_models.py"
        fi
        $verifier $mcn_model_path $output_dir $feat_dir
    fi
}

for mcn_model in "${model_list[@]}"
do
    mcn_model_path="${mcn_import_dir}/${mcn_model}.mat"
    convert_model $mcn_model_path
done
