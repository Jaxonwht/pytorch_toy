#!/usr/bin/env bash
#SBATCH --mem=10G
#SBATCH -p gpu-common

source activate pytorch
python word2vec/word_to_vec.py