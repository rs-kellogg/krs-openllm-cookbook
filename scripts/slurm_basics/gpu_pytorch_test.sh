#!/bin/bash

#SBATCH --account=e32337
#SBATCH --partition gengpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time 0:30:00
#SBATCH --mem=40G

module purge
module load python
source activate /kellogg/software/envs/gpu-pytorch
python pytorch_gpu_test.py
