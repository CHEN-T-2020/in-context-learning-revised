#!/bin/bash -l

#SBATCH --job-name=my_job           # Job name
#SBATCH --ntasks=1                  # Run on a single CPU

#SBATCH --gres=gpu:2080:1           # Request 1 GPU


#SBATCH --time=48:00:00            # Time limit hrs:min:sec
#SBATCH --output=slurm_output/%j.log # Output and error log path

# Load required modules (if any)
# module load <module_name>

source ~/.bashrc
source activate /home/tianqi/anaconda3/envs/in-context-learning

# Task 1
echo "Running Task:14Sep_skew_100_lstm_20dim_5layer_512_lr0.0001_dropout0.1_curriculumFalse_p_embeddingFalse_layernormFalse Time:2023-09-15 02:04:28"
python train.py --config conf/cust_models/14Sep_skew_100_lstm_20dim_5layer_512_lr0.0001_dropout0.1_curriculumFalse_p_embeddingFalse_layernormFalse.yaml



