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
use_ipython=true
mcn_import_dir="~/data/models/matconvnet"
output_dir="~/data/models/pytorch/mcn_imports"
verbose=false

# Convert ReLU modules to run "in place"
inplace=true

# enforce python2 usage (useful for ensuring backwards compatibility)
enforce_python2=false

# verification options
debug_mode=false # adds additional layers for clarity and verfies features
feat_dir="~/data/pt/pytorch-mcn/feats"

# Declare list of models to be imported (comment/uncomment selection to run)

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

#declare -a model_list=("imagenet-vgg-f-dag relu6")
#
## face descriptors
declare -a model_list=("vgg_face-dag")
#
## face descriptors
declare -a model_list=(
"senet50_scratch-dag"
"resnet50_scratch-dag"
"resnet50_ft-dag"
"senet50_ft-dag"
)
#"resnet50_scratch-dag"

#"se50_128D-ft"
declare -a model_list=(
"res50_128D-ft"
)

# imagenet models
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

# Emotion recognition models
# NOTE: For some models, we flatten after relu7 to avoid flattening before a
# batch norm layer.
declare -a model_list=("alexnet-face-fer-bn-dag relu7"
					   "vgg-m-face-bn-fer-dag relu7"
					   "vgg-vd-face-fer-dag relu6"
					   "vgg-vd-face-sfew-dag relu6"
					   "resnet50-face-sfew-dag prediction_avg"
                       )

declare -a model_list=("resnet50-ferplus-dag pool5_7x7_s1"
					   "senet50-ferplus-dag pool5_7x7_s1"
                       )
# dev
#declare -a model_list=("resnet50-ferplus-dag pool5_7x7_s1")
declare -a model_list=("vgg_face-dag pool5")

pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

function convert_model()
{
    mcn_model_path=$1
    flatten_layer=$2
	echo "Exporting MatConvNet model to PyTorch (may take some time)..."

    if [ $enforce_python2 = true ] ; then
        if [ $use_ipython = true ] ; then
            binary="ipython2"
        else
            binary="python2"
        fi
    else
        if [ $use_ipython = true ] ; then
            binary="ipython3"
        else
            binary="python3"
        fi
    fi
    converter="${binary} $SCRIPTPATH/python/importer.py --"
    if [ $refresh_models = true ] ; then
        opts="--refresh"
    fi
    if [ $debug_mode = true ] ; then
        opts="$opts --debug_mode"
    fi
    if [ $verbose = true ] ; then
        opts="$opts --verbose"
    fi
    if [ $inplace = true ] ; then
        opts="$opts --inplace"
    fi
    if [ ! -z "$flatten_layer" ] ; then
        opts="$opts --flatten_layer ${flatten_layer}"
    fi
    $converter $mcn_model_path $output_dir $opts

    if [ $debug_mode = true ] ; then
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
