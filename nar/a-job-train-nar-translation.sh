#!/bin/bash

#$ -l rt_AF=1
#$ -l h_rt=10:00:00
#$ -t 1
#$ -j y
#$ -cwd
#$ -m ea

source /etc/profile.d/modules.sh

module load gcc/11.2.0
module load python/3.10/3.10.4
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1

source ~/venv/af_pytorch/bin/activate

export OMP_NUM_THREADS=1
export NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

export HF_DATASETS_CACHE="/scratch/aae15163zd/cache/huggingface/datasets"

python_cmd="train-nar-translation.py"

# launch on slave nodes
node_rank=1
for slave_node in `cat $SGE_JOB_HOSTLIST | awk 'NR != 1 { print }'`; do
    qrsh -inherit -V -cwd $slave_node \
    eval "torchrun --nproc_per_node $NUM_GPUS_PER_NODE --nnodes $NHOSTS --node_rank $node_rank --master_addr `hostname` "$python_cmd &
    node_rank=`expr $node_rank + 1`
done

# launch on master node
node_rank=0
eval "torchrun --nproc_per_node $NUM_GPUS_PER_NODE --nnodes $NHOSTS --node_rank $node_rank --master_addr `hostname` "$python_cmd
