#!/bin/bash -l

#$ -P ec523
#$ -l h_rt=24:00:00
#$ -m beas
#$ -N lse_ducknet_uf1_train
#$ -j y
#$ -o save/LSEducknet_34_uf1/train.logs
#$ -pe omp 4
#$ -l gpus=2
#$ -l gpu_c=8.0

module load miniconda
conda activate /projectnb/ec523/projects/Team_A+/dl_prj_env
python3 /projectnb/ec523/projects/Team_A+/larynx_transfer_learning/github_branch/main.py --config lseducknet_uf1_config
