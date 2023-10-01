# MCryGAN

# Toolbox for machine learning on molecular crystals

## Development from archived public [version](https://github.com/InfluenceFunctional/mcrytools_Nov_2022)


## Reference
#TODO If you use this code in any future publications, please cite our work using

## Table of contents
- [Installation](#1-installation)
- [Dataset Management](#2-dataset-management)
- [Tools](3-tools)
- [Workflows](#4-running-a-job)
  - [Dataset Generation](#dataset-generation)
  - [Model Training](#model-training)
  - [Crystal Search & Optimization](#crystal-search)
- [Automated tests](#5-automated-tests)


## 1. Installation
1. Download the code from this repository via 

    `git clone git@github.com:InfluenceFunctional/MCryGan.git mcrygan`

2. Locate `molcrytools-env.yml` in the downloaded MCryGAN codebase directory. Edit to correspond to your CUDA version.
Then at the codebase directory, run 
3. 
    `conda env create -f molcrytools-env.yml` 

    on command line to create a conda virutal environment with most of the required packages.
3. Next, do

   `pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.htm`
    
    with your pytorch and cuda versions substituted in the pyg link. It is important that your cuda version, pytorch 
cuda version, and torch_geometric cuda version all match see 
[this link](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for more details.


The `molcrytools-env` environment should be activated when the installation script finishes, 
which means a string '(molcrytools-env)' should show up at the beginning of the command line prompt.

3. Login to your weights and biases account (necessary for run monitoring and reporting) with `wandb login` in the 
linux/Windows terminal
4. In configs/users create a .yaml file for yourself and edit the paths and wandb details to correspond to your preferences.
When running the code, use the following command line prompt, and the code will accept your configs. 
> --user YOUR_USERNAME

## 2. Dataset Management
If working on a shared system with access to a pre-generated molecular crystal dataset, skip this step.

This software may use crystal structures drawn from the Cambridge Structural Database (CSD) or 
collections of .cif files. It uses the ccdc Python API and RDKit packages to extract the structures and featurize them 
for training or evaluation. In training, we store crystals with a custom `crystaldata` structure, and 
parameterization scheme. We use the features listed in #TODO. If you would like to use a different
feature set, you may edit the featurizer #TODO, or, if your feature is fast to compute, add it at runtime in 
`dataset_management/modelling_utils.py`.

# TODO rewrite
* Set arguments in #todo `build_dataset.py` for where to construct the dataset, and whether to 
draw samples from the CSD or a (optionally nested) directory of .cif files,
and run `python build_dataset.py`. This takes #TODO hours to complete on a standard laptop, and processes
all the crystal samples into a featurized Pandas dataframe
   * Our working philosophy at this step is to only filter out structures which strictly cannot be used (e.g., missing
   atoms), and to filter down to desired crystal types (e.g., size, space group) at runtime.

* At runtime, there are a variety of options for how to curate the dataset. See your `.yaml` config for more details.

## 3. Tools

This toolkit uses a set of key tools in its workflows. 

