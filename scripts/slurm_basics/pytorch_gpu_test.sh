#!/bin/bash

#SBATCH --account=e32337
#SBATCH --partition gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time 0:30:00
#SBATCH --mem=40G

module purge all
module use --append /kellogg/software/Modules/modulefiles
module load micromamba/latest
source /kellogg/software/Modules/modulefiles/micromamba/load_hook.sh
micromamba activate /kellogg/software/envs/llm-test-env
python pytorch_gpu_test.py
