![image](https://github.com/InfluenceFunctional/MXtalTools/assets/30198118/ecc49717-b9b4-4901-9b59-8e4c8b919813)
# MXtalTools: Toolbox for machine learning on molecular crystals

## Documentation
See our detailed documentation including installation and deployment instructions at our [readthedocs](https://mxtaltools.readthedocs.io/en/latest/index.html) page.

## Usage
This package contains models and tools to assist in the training of models for a variety of tasks on molecular crystals, including most importantly:
1. Fast, parallel, differentiable, reproducible building of unit cells / supercells given molecule structure & crystal parameters.
2. Likewise, tools for analysis of generated structures.
3. Utilities for the collation and analysis of crystal & molecule datasets.
4. Flexible support for training & evaluating various types of models. 
5. Custom graph neural network models for molecular crystal learning tasks. 


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
7`standalone` - tools for true standalone deployment of crystal models, e.g., stability score & density prediction

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

