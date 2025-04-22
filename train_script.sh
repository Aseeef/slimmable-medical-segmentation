#!/bin/bash -l

#$ -P ec523
#$ -l h_rt=48:00:00
#$ -m beas
#$ -N us_ducknet_train
#$ -j y
#$ -o slim_train.logs
#$ -pe omp 4
#$ -l gpus=2
#$ -l gpu_c=8.0
#$ -l gpu_memory=24G

module load miniconda
cd /projectnb/ec523/projects/Team_A+/David_slimmable_net
conda activate /projectnb/ec523/projects/Team_A+/dl_prj_env
#python3 ./main.py --config slimducknetconfig       # For regular slimmable DUCKNet
python3 main.py --config usslimducknetconfig         # For US DUCKNet


