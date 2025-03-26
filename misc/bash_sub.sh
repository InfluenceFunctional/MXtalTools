#!/bin/bash
#SBATCH --time=0-04:00:00  # 4 hrs
#SBATCH --gres=gpu:1  # 1 GPU
#SBATCH --mem=20000M  # 20 GB RAM
#SBATCH --cpus-per-task=1  # 1 CPU Core
#SBATCH --mail-user=YOUR_EMAIL_ADDRESS
#SBATCH --mail-type=END
#SBATCH --array=0-2  # job indexes to run

source activate YOUR_PYTHON_ENV

python main.py --user YOUR_USERNAME --yaml_config /PATH_TO_EXPERIMENT_CONFIGS/experiment_$SLURM_ARRAY_TASK_ID.yaml  # runs experiment_0.yaml --> experiment_2.yaml
