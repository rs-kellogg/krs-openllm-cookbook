#!/bin/bash

#SBATCH -A your_quest_allocation_account
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0:30:00
#SBATCH --mem=40G

module purge
module use modulefiles
module load llama_cpp/2.38
python3 /code/gemma_test.py
