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
python train.py --config conf/cust_models/4Aug_lstm_3dim_20layer_256_lr1e-4_dropout0.1_positionalEmbedding.yaml



