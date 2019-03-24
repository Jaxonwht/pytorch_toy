#!/usr/bin/env bash
#SBATCH --mem=10G
#SBATCH -p gpu-common

source activate pytorch_old
python -u variational_autoencoder_comp.py