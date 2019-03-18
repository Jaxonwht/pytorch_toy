#!/usr/bin/env bash
#SBATCH --mem=20G
#SBATCH -p gpu-common

source activate pytorch
python word_to_vec.py