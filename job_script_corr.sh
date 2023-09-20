#!/bin/bash
#SBATCH --gres=gpu:2080:1           # Request 1 GPU
#SBATCH --time=4:00:00            # Time limit hrs:min:sec
#SBATCH --job-name=eval_corr           # Job name
#SBATCH --output=slurm_output/%j.log # Output and error log path

source ~/.bashrc
source activate /home/tianqi/anaconda3/envs/in-context-learning


# Task 1
echo "Running Task 1"
python eval_corr.py
