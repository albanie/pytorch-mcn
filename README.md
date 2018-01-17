pytorch-mcn
---

A tool for importing trained MatConvNet models into Pytorch (if you were hoping to go the other way, try [mcnPyTorch](https://github.com/albanie/mcnPyTorch)).

### Demo

To run the importer, set the path to the MatConvNet model and supply an output directory (where the imported PyTorch models will be stored) in the `importer.sh` script.  There are a few examples in the script which can be commented/uncommented to run as a demo.  Then run `bash importer.sh`.

### Imported Models

A number of standard models have been imported and verified, and can be found [here](http://www.robots.ox.ac.uk/~albanie/pytorch-models.html).

### Verification

Verifying an imported model requires MATLAB and a copy of MatConvNet (the specific dependencies are given in `compare/startup.m`).  The process is as follows:

1. Run the `compare/featureDumper.m` script to dump intermediate features from the original MatConvNet model to disk.
2. Import model to PyTorch in `debug_mode` (an option that can be set in `importer.sh`.  This will generate additional source code in the PyTorch model definition that stores every intermediate tensor computed by the network.
3. Run the `compare/compare_models.py` script, which will perform a numerical comparison between the tensors.


### Notes

Model conversion between frameworks can be challenging, because the layers and modules of each framework do not have an exact correspondence. As a result, there is often a bit of work involved in the conversion process, particularly for non-standard architectures.

This tool requires at least Python 3.5 and PyTorch 0.3.0 (by default, `ipython` will be used, but you can switch to standard python by changing a config variable in `importer.sh`).  The tool expects the MatConvNet models to be in `dagnn` format (the [ensure_dags.m](compare/ensure_dags.m) script converts models to this format from SimpleNN if required).

Ideally in future the model conversion process will run via onnx but it seems that currently quite a lot of support is missing for required functionality.  The plan is therefore to update the converter when possible.
