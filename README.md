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
1. Download the code from this repository via git clone #TODO (API key instructions)
2. Locate `macos_installation.sh` in the downloaded MCryGAN codebase directory. Edit lines XX to accurately 
represent your system's CUDA version (if any). 
Then at the codebase directory, run `$ source macos_installation.sh` in command line to create a conda 
virtual environment named `mcrygan` and install required dependencies. 
The `mcrygan` environment should be activated when the installation script finishes, 
which means a string '(mcrygan)' should show up at the beginning of the command line prompt.
3. Login to your weights and biases account (necessary for run monitoring and reporting) with `wandb login` in the 
linux terminal
4. Edit the default paths and wandb details in `main.py` to correspond to your preferences.
#TODO greene specific instructions

## 2. Dataset Management
If working on a shared system with access to a pre-generated molecular crystal dataset, skip this step.

This software may use crystal structures drawn from the Cambridge Structural Database (CSD) or 
collections of .cif files. It uses the #todo packages the extract the structures and featurize them
for training purposes. In training, we store crystals with a custom `crystaldata` structure, and 
minimal parameterization scheme. We use the features listed in #TODO. If you would like to use a different
feature set, you may edit the featurizer #TODO, or, if your feature is fast compute, add it at runtime.

* Set arguments in #todo `build_dataset.py` for where to construct the dataset, and whether to 
draw samples from the CSD or a (optionally nested) directory of .cif files,
and run `python build_dataset.py`. This takes #TODO hours to complete on a standard laptop, and processes
all the crystal samples into a featurized Pandas dataframe
   * Our working philosophy at this step is to only filter out structures which strictly cannot be used (e.g., missing
   atoms), and to filter down to desired crystal types (e.g., size, space group) at runtime.

* At runtime, there are a variety of options for how to curate the dataset. See your `.yaml` config for more details.
* One may also add or remove features from training samples in `dataset_management/utils.py`. 

## 3. Tools

This toolkit uses a set of key tools in its workflows. 

* `crystaldata` object
* Crystal analysis
* Crystal building
* Crystal modelling
* Crystal search
* Reporting