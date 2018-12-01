#!/bin/bash
#SBATCH --mem=10G
#SBATCH -p gpu-common

source activate opennmt
python -u word2vec.py
