#!/bin/bash -l

#$ -P ec523
#$ -l h_rt=24:00:00
#$ -m beas
#$ -N ducknet_train
#$ -j y
#$ -o train.logs
#$ -pe omp 2
#$ -l gpus=3
#$ -l gpu_c=7.0

module load miniconda
conda activate /projectnb/ec523/projects/Team_A+/dl_prj_env
python3 ./main.py

