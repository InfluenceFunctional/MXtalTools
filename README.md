![image](https://github.com/InfluenceFunctional/MXtalTools/assets/30198118/f63d672b-6145-4bb8-8951-b9aad0a20650)
# MXtalTools
# Toolbox for machine learning on molecular crystals

## Archived version for classification of bulk molecule polymorphs


## 1. Installation
1. Download the main repository via
    ```bash
    git clone git@github.com:InfluenceFunctional/MXtalTools.git MXtalTools
    ```
    then checkout the classifier branch.
    ```bash
    git checkout mol_classifier
    ```
   In future, the mol_classifier feature set will be incorporated into the main branch, and generalized to other molecules. 
   Until then, see the relevant scripts in the `bulk_molecule_classification` directory.
2. Locate `OVITO_torch.yaml` in `installation_scripts`. Edit to correspond to your CUDA version. Then run
    ```bash
    conda env create -f OVITO_torch.yaml
    ```
   on command line to create a conda virutal environment with the required packages.
   Note that installation may take some time, particularly due to OVITO.
   Users may have to install packages separately one-by-one.
   If OVITO is a sufficiently large issue, it can be omitted but then classified trajectories cannot be saved. 

   Activate the environment with 
    ```bash
    conda activate OVITO_torch
    ```
3. Login to your weights and biases ("wandb") account, which is necessary for run monitoring and reporting with 
    ```bash
    wandb login
    ```
   on the terminal command line.
   You may have to provide your login info / api key if you have not already logged into wandb on the current machine.

## 2. Configs
- Configs are loaded into `main_classifier.py` from this python script, which controls all behavior of the pipeline. 
- Two possibilities:
  1. Load a single config, here named `dev` and execute.
  2. If an integer 'run_num' is passed via command line, e.g., via
  ```bash
  python classifier_main.py --config run_num
  ```
  `classifier_main.py` will execute run number run_num from the list of configs defined as `configs` in `dev_configs.py`.
    This is most useful when executing a batch of jobs e.g., on cluster compute. 
- Key config items are
  - run_name: if 'nic' or 'urea' in name, will use relevant hardcoded parameters. Otherwise will fail.  
  - The various model hyperparamters: Note, if loading a pre-trained model, must set these as the same as in the model you are trying to load.
    Note that batch_size here is synthesized by gradient accumulation. 
    The model always passes forward one snapshot at a time.
  - Mode options, train_model, do_classifier_evaluation, and trajs_to_analyze_list (will not execute if list is None).
  - classifier_path: path to pre-trained model
  - datasets_path: where to store the datasets generated from dumps processing
  - dataset_name: name of the dataset to be loaded or generated (if not already built), composed of the dumps set in dumps_dirs'.
  - dumps_path: where to look for dumps directories with tha names given in dumps_dirs.
  - runs_path: where to store checkpoints, mid-run artifacts.
  - results_path: where to save analysis outputs, figures, and other artifacts.

## 3. Datasets
This software generates training datasets of LAMMPS MD simulations from collections of LAMMPS dump files.
The `generate_dataset_from_dumps` method processes dumps into Pandas data frames, which can be used for training.   
This method is somewhat slow, so its outputs are saved to a local directory to save time in future runs. 
Several dumps can be processed at a time by passing a list of directories 'dumps_dirs', which will all be searched up to one level deep for `.dump` files, and then collected into a single combined dataset.
To gather the parameters of the run, one must manually supply certain information about the run, or if the trajectory was generated with our automated LAMMPS script, the `run_config.npy` is automatically loaded and processed. 
Collation includes filtering of structures which are somehow invalid. 
Current processing tools are hardcoded for either nicotinamide or urea, but can be easily extended. 
Note that dumps files must be appropriately formatted for the collation function `process_dump` to work.

At runtime, training data is processed into train & test datasets, with highly flexible options for custom filtering and construction. 
This is also where topologies such as 'surface' and 'bulk' are currently set.
See `collect_to_traj_dataloaders` for details on such options. 
Note that most depend on accurate knowledge of the run parameters, gathered in the above dumps processing stage. 


## 4. Modes
There are multiple modes for training, inference and analysis. 

1. train_classifier
   - Here we pass trajectory snapshots to the classifier one-at-a-time, and evaluation classification loss, then backpropagate and repeat in the usual way. 
   - Several metrics are logged to wandb, including separate losses for polymorph classification vs. sample topology. 
2. classifier_evaluation
   - For evaluating pre-trained models on given datasets. 
   - Prepares detailed evaluation metrics and figures, and logs to wandb.
3. trajectory_analysis
   - Analyzes a given LAMMPS dump file using a pre-trained classifier. 
   - Saves results to a local dict (warning, can be very large!), generates analysis figures, and logs to wandb. 
   - Note analysis is somewhat hardcoded to urea or nicotinamide. 
   - The `write_ovito` call at the end of this function is the only place OVITO is used here, and can be skipped if there are issues with OVITO installation. 

## 5. Key Components
1. `classifier_main.py`: main file controlling I/O and calling workflows.
2. `mol_classifier.py`: graph neural network for analyzing MD snapshots.
3. `dataset_prep.py`: process and generate training datasets.
4. `workflows.py`: scripts and utilities for the above-noted modes. 
5. `dev_configs.py`: config file for controlling `classifier_main.py`

## 6. Bash Utilities
The below is a standard script for submitting jobs on a system with the slurm scheduler.
The given script runs configs 0-47 from `dev_configs.py` for up to 6 hours on 2 CPUs with 20GB of RAM, and will email the user when the job finishes or crashes. 
```bash
#!/bin/bash
#SBATCH --time=0-06:00:00
#SBATCH --mem=20000M
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=your_email_address
#SBATCH --mail-type=END
#SBATCH --array=0-47

source activate your_python_environment

cp bulk_molecule_classification/classifier_main.py ./
python classifier_main.py --config $SLURM_ARRAY_TASK_ID
rm classifier_main.py
```
