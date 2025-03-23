#!/bin/bash -l

#$ -P ec523
#$ -l h_rt=24:00:00
#$ -m beas
#$ -N ducknet_train
#$ -j y
#$ -o train.logs
#$ -pe omp 2
#$ -l gpus=2
#$ -l gpu_type=V100

module load miniconda
conda activate my_env
python3 ./main.py

