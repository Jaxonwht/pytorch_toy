#!/usr/bin/env bash
#SBATCH --mem=8G
#SBATCH -p gpu-common

source activate pytorch_old
python -u word_to_vec_comp.py