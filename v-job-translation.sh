#!/bin/bash

#$ -l rt_F=2
#$ -l h_rt=20:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
source ~/venv/pytorch/bin/activate
module load gcc/9.3.0
module load python/3.8/3.8.13
module load cuda/11.1/11.1.1
module load cudnn/8.0/8.0.5
module load nccl/2.8/2.8.4-1

export OMP_NUM_THREADS=1
export NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export HF_DATASETS_CACHE="/scratch/aae15163zd/cache/huggingface/datasets"

python_cmd="train-translation.py"

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

# finalize
wait
exit 0
