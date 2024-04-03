#!/bin/bash

#SBATCH -A your_quest_allocation_account
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:30:00
#SBATCH --mem=40G


module purge
module load mamba/23.1.0
source /hpc/software/mamba/23.1.0/etc/profile.d/conda.sh
source activate /kellogg/software/envs/gpu-llama2

python proc_llamaindex_embed_e5_short.py

