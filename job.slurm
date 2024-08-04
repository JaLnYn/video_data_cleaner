#!/bin/bash

#SBATCH --job-name=python-training
#SBATCH --output=slurm_out/result-%J.out
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a40:1

# Load Python and any other modules required for your job
# module load python/3.10

# Activate your virtual environment
source ../talkinghead/venv/bin/activate

# Execute the Python script with the config path provided from the command line
python create_list.py

# Deactivate the virtual environment
deactivate

