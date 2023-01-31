#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd
#$ -m ea
#$ -M gsohrab4386@gmail.com
#source ~/miniconda3/bin/activate ddp-optimus-v2
source ~/miniconda3/bin/activate
conda activate conda-env-seq2seq-eval
source /etc/profile.d/modules.sh
module load gcc/9.3.0
#module load cmake/3.19
#module load intel-mkl/2020.0.4
#module load openjdk/11.0.6.10
module load python/3.8/3.8.13
#module load openmpi/4.0.5
#module load cuda/10.2/10.2.89
module load cuda/11.2/11.2.2
module load cudnn/8.1/8.1.1
#module load cudnn/8.0/8.0.5
module load nccl/2.8/2.8.4-1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
#HOME=$PWD
DATA_DIR="../data/datasets/glue_data"
#MODEL_DIRS="../results/wiki/checkpoint-220000"
#RESULT_DIR="../results/glue/b2b-base-cased/"

export TASK_NAME=QNLI
export OPTIMIZER=AdamW

#SEEDS="66 90 99 168 786"
#LEARNING_RATES="2e-5 3e-5 4e-5 5e-5"

#MODEL_DIRS="../results/wiki/encode_vae_decoder/checkpoint-encoder-700000/ ../results/wiki/encode_vae_decoder/checkpoint-decoder-700000/"
#RESULT_DIR="../results/glue/b2b-encoder-vae-decoder/"

MODEL_DIRS="../../repo/BERT2BERT-transformers-upgrade/output_dir/encoder_decoder/checkpoint-4000/"
RESULT_DIR="../results/glue/b2b-encoder-decoder/"

SEEDS="99"
LEARNING_RATES="5e-5"

for MODEL_DIR in $MODEL_DIRS; do
    for SEED in $SEEDS; do
	for LEARNING_RATE in $LEARNING_RATES; do
	    python ./examples/run_glue_adamw.py \
		--model_name_or_path $MODEL_DIR \
		--model_type bert \
		--task_name $TASK_NAME \
		--do_train \
		--do_eval \
		--max_seq_length 128 \
		--per_gpu_train_batch_size 24 \
		--learning_rate $LEARNING_RATE \
		--num_train_epochs 3 \
		--output_dir $RESULT_DIR/$TASK_NAME/ \
		--data_dir $DATA_DIR/$TASK_NAME/ \
		--overwrite_output_dir \
		--eval_all_checkpoints \
		--log_path $RESULT_DIR/$TASK_NAME/log_$TASK_NAME-lr_$LEARNING_RATE-optim_$OPTIMIZER.txt \
		--seed $SEED \
		--logging_steps 4000 \
		--save_steps 4000 
	done
    done
done
