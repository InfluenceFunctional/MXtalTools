# MXtalTools

# Toolbox for machine learning on molecular crystals

## Development from archived public [version](https://github.com/InfluenceFunctional/mcrytools_Nov_2022)


## Reference
TODO If you use this code in any future publications, please cite our work using

## Table of contents 
TODO write ToC

## 1. Installation
1. Download the code from this repository via

    todo confirm path names
```bash
git clone git@github.com:InfluenceFunctional/MXtalTools.git MXtalTools
```

2. Locate `mxtaltools-env.yaml` in the downloaded MXtalTools codebase directory. Edit to correspond to your CUDA version. Then at the codebase directory, run
```bash
conda env create -f mxtaltools-env.yaml
```
   on command line to create a conda virutal environment with the required packages.
   Activate the environment with 
```bash
conda activate mxtaltools-env
```
3. Login to your weights and biases ("wandb") account, which is necessary for run monitoring and reporting with 
```bash
wandb login
```

   on the terminal command line.
4. In configs/users create a .yaml file for yourself and edit the paths and wandb details to correspond to your preferences.
When running the code, append the following to your command line prompt. 
    > --user YOUR_USERNAME


## 2. Datasets
1. This software generates training datasets of molecular crystal structures from collections of .cif files.
    .cifs are collated and processed primarily with the CSD Python API and RDKit.
    Collation includes filtering of structures which are somehow invalid. 
    Invalid conditions include: symmetry features disagree, no atoms in the crystal, RDKit rejects the structure outright. 
    The Cambridge Structural Database (CSD) can be processed by first dumping it to .cif files, or directly with minor modifications.
    Customized functions are available for processing CSD Blind Test submissions TODO clean & test.
    
2. In the most common case, processing the CSD, to generate a dataset, run the following scripts,
    `dump_csd.py` --> `cif_processor.py` --> `manager.py`,
    with the appropriate paths set in each script.
    `cif_processor.py` takes on the order of dozens of hours to process the full CSD (>1M crystals).
    `manager.py` also may take a few minutes to process a large dataset, as this is where we do pose analysis, 
    duplicates search, and some indexing tasks.
    We recommend running several instances in parallel to reduce this time.
    As they process datasets chunkwise in random order, this parallelism is fairly efficient.
    Note that the speed here depends strongly on disk read-write speed. 

3. The list of featurzed computed for each atom/molecule/crystal are listed in TODO generate list.
If you would like to use a different feature set, you may edit `dataset_management/featurization_utils.py`, 
or, if your features are fast to compute, add them at runtime in `dataset_management/modelling_utils.py`.

4. Subsets of features, can be selected in the dataset config used for a given run in `configs/dataset`. 
Likewise, filters can be applied to the dataset to limit the crystals included. 
Such filters may commonly include: molecule size, crystal setting, atomic number. 

5. We also have functions for identifying duplicate molecules and polymorphs of the same molecule. 
When filtering these, we identify all the duplicates and pick a single 'representative' sample at random.


## 3. Modes
There are multiple modes for training, inference and analysis. 

1. train_crystal_models

    Within training mode, there are two types of models which can currently be trained. 
   1. regression mode - for regressing some property against single-molecule information
   2. gan mode - for training generator or discriminator models, not necessarily as a GAN per-se. 
   Done this way because of significant overlap in code and reporting. 
   3. Molecule autoencoder and normalizing flow models - DEPRECATED
2. model_inference WIP - incomplete
   - For evaluating pre-trained models on given datasets. Currently folded into training mode with max_epochs=0
3. crystal_structure_prediction WIP - incomplete
   - For doing a CSP search over one or more molecules/crystals, including sample generating, clustering and optimization. 
   Currently incomplete, and folded into training mode.
4. embedding mode WIP - rebuilding
   - For using one of our crystal models (so far, discriminator) as an embedding generator. 

Modes and datasets can be called via experiment configs and dataset configs.
- For experiment specification (usually in `configs/experiments` and called on the command line or in the user config)
see `configs/test_configs/discriminator.yaml`.
This loads the full_gan specified dataset, and trains a discriminator model only on randn and distorted fakes.
- For dataset construction, see the configs `configs/dataset/full_gan.yaml` and `configs/dataset/skinny_regression.yaml`.
These construct training datasets for training generator/discriminator models with all atom & molecule features, and
a regressor model with only a few input features. 
Note the differences in filter conditions between modes, such as the filtering of all polymorphs and duplicate molecules
in the regression task, and the filtering of dubious or nonstandard settings in the generation task. 
- For an example slurm script for cluster batch submission, see `bash_sub.sh`.

## 4. Components

1. crystal_modeller - class which contains everything else and does all the work
2. logger - handles training statistics and reporting to weights and biases
3. crystal builder - generates supercells / molecule clusters given molecule & symmetry information for training and reporting
4. molecule_graph_model - wrapper for GraphNeuralNetwork which parses i/o according to the various needs of different types of models
5. configs
   1. users - path and wandb login info for separate users
   2. dataset - specifies information for dataset construction and featurization
   3. main / dev / experiments - define all other parameters of a given run including losses, hyperparameters, convergence, etc.
6. dataset_management - tools for dataset generator, curation, and modelling
7. csp / sampling - WIP standalone modules for crystal structure prediction
8. standalone - tools for true standalone deployment of crystal models, e.g., as score model for training external models