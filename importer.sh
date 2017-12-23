# importer.sh
# Import matconvnet models into pytorch
#
# --------------------------------------------------------
# pytorch-mcn
# Licensed under The MIT License [see LICENSE.md for details]
# Copyright (C) 2017 Samuel Albanie
# --------------------------------------------------------

# set paths/options
mcn_import_dir="~/data/models/matconvnet"
output_dir="~/data/models/pytorch/mcn-imports"
refresh_models=true

pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

function convert_model()
{
    mcn_model_path=$1
	converter="ipython $SCRIPTPATH/python/importer.py --"
	mkdir -p "${output_dir}"

	echo "Exporting MatConvNet model to PyTorch (may take some time) ..."
    $converter --refresh=$refresh $mcn_model_path $output_dir
}

# Example models from the torchvision module
#declare -a model_list=("alexnet" "vgg11" "vgg13" "vgg16" "vgg19" \
                       #"squeezenet1_0" "squeezenet1_1" "resnet152" )

## alternatively, specify a custom model (will require modifications to
## the python source to add support for unrecognised layers)
#declare -a model_list=("resnext_101_32x4d")
#model_def="${HOME}/.torch/models/resnext_101_32x4d.py"
#weights="${HOME}/.torch/models/resnext_101_32x4d.pth"

declare -a model_list=("squeezenet1_0")

for mcn_model in "${model_list[@]}"
do
    mcn_model_path="${mcn_import_dir}/${mcn_model}.mat"
    convert_model $mcn_model_path
done
