#!/bin/bash -l

#$ -P ec523
#$ -l h_rt=24:00:00
#$ -m beas
#$ -N ducknet_train
#$ -j y
#$ -o slim_train.logs
#$ -pe omp 4
#$ -l gpus=2
#$ -l gpu_c=8.0
#$ -l gpu_memory=24G

module load miniconda
conda activate /projectnb/ec523/projects/Team_A+/dl_prj_env
#python3 ./main.py --config slimducknetconfig       # For regular slimmable DUCKNet
python main.py --config usslimducknetconfig         # For US DUCKNet


