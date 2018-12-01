#!/bin/bash
#SBATCH --mem=10G
#SBATCH -p gpu-common

source activate opennmt
python word2vec.py