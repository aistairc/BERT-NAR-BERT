#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=1:00:00
#$ -t 1:2
#$ -j y
#$ -cwd
#$ -v GPU_COMPUTE_MODE=1
#$ -m ea

source /etc/profile.d/modules.sh                                                                                                                            
module load gcc/9.3.0                                                                                                                                       
module load python/3.8/3.8.13                                                                                                                               
module load cuda/11.1/11.1.1                                                                                                                                
module load cudnn/8.0/8.0.5                                                                                                                                 
module load openmpi/4.0.5
module load nccl/2.8/2.8.4-1                                                                                                                                

source ~/venv/pytorch/bin/activate                                                                                                                          

CACHE_DIR=".abci_ddp_caches/$JOB_ID"
mkdir -p "$CACHE_DIR"

sleep 10
mpirun -np 1 cp "$SGE_JOB_HOSTLIST" "$CACHE_DIR/$SGE_TASK_ID"
sleep 10

MASTER_FILE="$CACHE_DIR/undefined"
if [ ! -f "$MASTER_FILE" ]; then
    MASTER_FILE="$CACHE_DIR/1"
fi

export NUM_NODES=$(ls -1 "$CACHE_DIR" | wc -l)
export NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(cat "$MASTER_FILE")
export NODE_RANK=$(expr $SGE_TASK_ID - 1)

mpirun -np 1 \
torchrun --nproc_per_node $NUM_GPUS_PER_NODE --nnodes $NUM_NODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port=29500 run_pretraining_ddp_b2b.py --batch_size 3 --fp16
