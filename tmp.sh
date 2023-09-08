#!/bin/bash

#SBATCH --output=slurm_output/%j.log # Output and error log path

source ~/.bashrc
source activate /home/tianqi/anaconda3/envs/in-context-learning

python train_linear.py --config conf/probing/linear_regression_probing_10.yaml