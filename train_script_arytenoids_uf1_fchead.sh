#!/bin/bash -l

#$ -P ec523
#$ -l h_rt=24:00:00
#$ -m beas
#$ -N arytenoid_ducknet_uf1_fchead_train
#$ -j y
#$ -o /projectnb/ec523/projects/Team_A+/larynx_transfer_learning/github_branch/save/arytenoidsducknet_34_uf1_fchead/train.logs
#$ -pe omp 4
#$ -l gpus=2
#$ -l gpu_c=8.0

echo "Saving to path: $SAVE_DIR"
ls -lah /projectnb/ec523/projects/Team_A+/larynx_transfer_learning/github_branch/save/arytenoidsducknet_34_uf1_fchead/
module load miniconda
conda activate /projectnb/ec523/projects/Team_A+/dl_prj_env
python3 /projectnb/ec523/projects/Team_A+/larynx_transfer_learning/github_branch/main.py --config arytenoidsducknet34_uf1_fchead_config

