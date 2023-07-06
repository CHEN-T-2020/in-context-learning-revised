#!/bin/bash -l

#SBATCH --job-name=my_job           # Job name
#SBATCH --ntasks=1                  # Run on a single CPU

#SBATCH --gres=gpu:2080:1                 # Request 1 GPU
#SBATCH --nodelist=ink-ellie

#SBATCH --time=48:00:00            # Time limit hrs:min:sec
#SBATCH --output=slurm_output/%j.log # Output and error log path

# Load required modules (if any)
# module load <module_name>

source ~/.bashrc
source activate /home/tianqi/anaconda3/envs/in-context-learning

# Task 1
echo "Running Task 1"
python train.py --config conf/cust_models/5July_gpt2_20dim_12layer_256_RL.yaml

# Task 2
echo "Running Task 2"
python train.py --config conf/cust_models/5July_mlp_20dim_22layer_1024_RL.yaml


