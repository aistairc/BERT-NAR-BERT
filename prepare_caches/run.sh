#!/bin/bash

#$ -l rt_C.large=1
#$ -l h_rt=2:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
source ~/venv/pytorch/bin/activate
module load gcc/8.5.0
module load python/3.10

export HF_DATASETS_CACHE="/scratch/aae15163zd/cache/huggingface/datasets"

python prepare_inputs_for_causal_LM.py
