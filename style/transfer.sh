#!/usr/bin/env bash
#SBATCH --mem=10G
#SBATCH -p gpu-common

source activate pytorch_old
python -u political_transfer_comp.py