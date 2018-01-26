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
debug_mode=false
use_ipython=true
mcn_import_dir="~/data/models/matconvnet"
output_dir="~/data/models/pytorch/mcn_imports"
verbose=false

# verification options
verify_model=false
feat_dir="~/data/pt/pytorch-mcn/feats"

# Declare list of models to be imported (comment/uncomment selection to run)
declare -a model_list=("squeezenet1_0-pt-mcn"
                       "squeezenet1_1-pt-mcn"
                       "alexnet-pt-mcn"
                       "vgg11-pt-mcn"
                       "vgg13-pt-mcn"
                       "vgg16-pt-mcn"
                       "vgg19-pt-mcn"
                       "resnet18-pt-mcn"
                       "resnet34-pt-mcn"
                       "resnet50-pt-mcn"
                       "resnet101-pt-mcn"
                       "resnet152-pt-mcn"
                       "inception_v3-pt-mcn"
                       "densenet121-pt-mcn"
                       "densenet161-pt-mcn"
                       "densenet169-pt-mcn"
                       "densenet201-pt-mcn")

# Examples of models that require the user to provide the location of a
# "flatten" operation (which corresponds to the pytorch module after which
# a View(x,-1) reshape will be performed).  This is passed as the second
# argument (for instance, `relu6` in the examples below).  See
# `python/importer.py` for more details of the importer interface.
declare -a model_list=("imagenet-matconvnet-vgg-f-dag relu6"
                       "imagenet-matconvnet-alex relu6"
                       "imagenet-matconvnet-vgg-m-dag relu6"
                       "imagenet-matconvnet-vgg-verydeep-16-dag relu6")

# NetVLAD feature extractors
declare -a model_list=("vd16_offtheshelf_conv5_3_max-dag"
                       "vd16_pitts30k_conv5_3_max-dag"
                       "vd16_tokyoTM_conv5_3_max-dag")

# Pedestrian Alignment Network
declare -a model_list=("ped_align")

# Emotion recognition models
declare -a model_list=("alexnet-face-bn-dag"
					   "alexnet-face-fer-bn-dag"
					   "vgg-m-face-bn-dag"
					   "vgg-vd-face-fer-dag"
					   "vgg-vd-face-sfew-dag"
					   "resnet50-face-bn-dag"
					   "resnet50-face-sfew-dag"
                       )

pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

function convert_model()
{
    mcn_model_path=$1
    flatten_layer=$2
	echo "Exporting MatConvNet model to PyTorch (may take some time)..."
    if [ $use_ipython = true ] ; then
        converter="ipython3 $SCRIPTPATH/python/importer.py --"
    else
        converter="python3 $SCRIPTPATH/python/importer.py"
    fi
    if [ $refresh_models = true ] ; then
        opts="--refresh"
    fi
    if [ $debug_mode = true ] ; then
        opts="$opts --debug_mode"
    fi
    if [ $verbose = true ] ; then
        opts="$opts --verbose"
    fi
    if [ ! -z "$flatten_layer" ] ; then
        opts="$opts --flatten_layer ${flatten_layer}"
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

for mcn_pair in "${model_list[@]}"
do
    tokens=($mcn_pair)
    mcn_model=${tokens[0]}
    flatten_layer=${tokens[1]}
    echo "importing $mcn_model"
    if [ ! -z "$flatten_layer" ] ; then
        echo "flattening at $flatten_layer"
    fi
    mcn_model_path="${mcn_import_dir}/${mcn_model}.mat"
    convert_model $mcn_model_path $flatten_layer
done
