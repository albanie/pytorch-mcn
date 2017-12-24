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
mcn_import_dir="~/data/models/matconvnet"
output_dir="~/data/models/pytorch/mcn-imports"

pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

# Examples (uncomment to run)
declare -a model_list=("squeezenet1_0-pt-mcn")

function convert_model()
{
    mcn_model_path=$1
	converter="ipython $SCRIPTPATH/python/importer.py --"
	mkdir -p "${output_dir}"
	echo "Exporting MatConvNet model to PyTorch (may take some time)..."
    if [ $refresh_models = true ] ; then
        opts="--refresh"
    fi
    if [ $debug_mode = true ] ; then
        opts="$opts --debug_mode"
    fi
    $converter $mcn_model_path $output_dir $opts
}

for mcn_model in "${model_list[@]}"
do
    mcn_model_path="${mcn_import_dir}/${mcn_model}.mat"
    convert_model $mcn_model_path
done
