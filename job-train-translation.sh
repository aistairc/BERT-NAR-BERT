#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=4:00:00
#$ -t 1-4
#$ -j y
#$ -cwd
#$ -v GPU_COMPUTE_MODE=1
#$ -m ea

source /home/aad13940yw/anaconda3/bin/activate b2b-gen

source /etc/profile.d/modules.sh

module load gcc/9.3.0
module load cmake/3.22.3
module load intel-mkl/2022.0.0
module load openjdk/11.0.15.0.9
module load python/3.8/3.8.13
module load openmpi/4.0.5
module load cuda/11.2/11.2.2
module load cudnn/8.1/8.1.1
module load nccl/2.8/2.8.4-1 

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0


CACHE_DIR=".abci_ddp_caches/$JOB_ID"

mkdir -p "$CACHE_DIR"

sleep 1

mpirun -np 1 cp "$SGE_JOB_HOSTLIST" "$CACHE_DIR/$SGE_TASK_ID"

sleep 1

MASTER_FILE="$CACHE_DIR/undefined"

if [ ! -f "$MASTER_FILE" ]; then
    MASTER_FILE="$CACHE_DIR/1"
fi

export NUM_NODES=$(ls -1 "$CACHE_DIR" | wc -l)
export NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(cat "$MASTER_FILE")
export NODE_RANK=$(expr $SGE_TASK_ID - 1)

export PYTHONFAULTHANDLER=1

echo "NUM_NODES=$NUM_NODES"
echo "NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "NODE_RANK=$NODE_RANK"


TOTAL_GPUS=$( expr ${NUM_NODES} '*' ${NUM_GPUS_PER_NODE})
echo "Total GPUS:" $TOTAL_GPUS


mpirun -np 1 \
    torchrun \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --master_addr="$MASTER_ADDR" \
    --master_port=29500 \
    ./training_vae_b2b_translation.py \

