![image](https://github.com/InfluenceFunctional/MXtalTools/assets/30198118/ecc49717-b9b4-4901-9b59-8e4c8b919813)
# MXtalTools: Toolbox for machine learning on molecular crystals

## Documentation
See our detailed documentation including installation and deployment instructions at our [readthedocs](https://mxtaltools.readthedocs.io/en/master/) page.

## Installation for Users


1. Install PyTorch, Pytorch Geometric (including torch-scatter, torch-sparse, torch-cluster),  based on your system and CUDA version:  
[PyTorch installation guide](https://pytorch.org/get-started/locally/)  
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

2. Install this package:

    ```bash
    pip install mxtaltools
    ```


## Installation for Developers

1. Download the code from this repository via

    ```bash
    git clone git@github.com:InfluenceFunctional/MXtalTools.git MXtalTools
    ```
2. Create a python environment of your choice. We recommend using pip+virtualenv. 
3. Install PyTorch, Pytorch Geometric (including torch-scatter, torch-sparse, torch-cluster),  based on your system and CUDA version:  
[PyTorch installation guide](https://pytorch.org/get-started/locally/)  
[PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
4. Install remaining requirements with 

    ```bash
    poetry install
    ```
5. If you plan to train any models, login to your weights and biases ("wandb") account, which is necessary for run monitoring and reporting with 
    
   ```bash
    wandb login
    ```
6. In configs/users create a .yaml file for yourself and edit the paths and wandb details to correspond to your preferences.
When running the code, append the following to your command line prompt. 
    
   ```
    --user YOUR_USERNAME
    ```
   
7. If you plan to construct crystal datasets from .cif files, you'll need to install the CSD python api, with a valid license from CCDC.

   [CSD Python API]([PyTorch installation guide](https://pytorch.org/get-started/locally/))

<!--
## 2. Datasets
1. This software generates training datasets of molecular crystal structures from collections of .cif files.
    .cifs are collated and processed primarily with the CSD Python API and RDKit.
    Collation includes filtering of structures which are somehow invalid. 
    Invalid conditions include: symmetry features disagree, no atoms in the crystal, RDKit rejects the structure outright. 
    The Cambridge Structural Database (CSD) can be processed by first dumping it to .cif files, or directly with minor modifications.
    Customized functions are available for processing CSD Blind Test submissions TODO clean & test.
    
2. In the most common case, processing the CSD, to generate a dataset, run the following scripts,
    `dump_csd.py` -> `cif_processor.py` -> `manager.py`,
    with the appropriate paths set in each script.
    `cif_processor.py` takes on the order of dozens of hours to process the full CSD (>1M crystals).
    `manager.py` also may take a few minutes to process a large dataset, as this is where we do pose analysis, 
    duplicates search, and some indexing tasks.
    We recommend running several instances in parallel to reduce this time.
    As they process datasets chunkwise in random order, this parallelism is fairly efficient.
    Note that the speed here depends strongly on disk read-write speed. 


### Key components
1. `crystal_modeller` - class which contains everything else and does all the work
2. `logger` - handles training statistics and reporting to weights and biases
3. `crystal_builder` - generates supercells / molecule clusters given molecule & symmetry information for training and reporting
4. `molecule_graph_model` - wrapper for GraphNeuralNetwork which parses i/o according to the various needs of different types of models
5. configs
   1. users - path and wandb login info for separate users
   2. dataset - specifies information for dataset construction and featurization
   3. main / dev / experiments - define all other parameters of a given run including losses, hyperparameters, convergence, etc.
6. `dataset_management` - tools for dataset generator, curation, and modelling
7. `standalone` - tools for true standalone deployment of crystal models, e.g., stability score & density prediction
-->

## Reference
If you use this code in any future publications, please cite our work using
```@article{kilgour2023geometric,
  title={Geometric deep learning for molecular crystal structure prediction},
  author={Kilgour, Michael and Rogal, Jutta and Tuckerman, Mark},
  journal={Journal of chemical theory and computation},
  volume={19},
  number={14},
  pages={4743--4756},
  year={2023},
  publisher={American Chemical Society}
}
```

