#!/bin/bash

set -x

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=20
export TOKENIZERS_PARALLELISM=False


DEV=1,2
PORT=1235
OUTPUT=$1
ZERO_STAGE=2
UNLEARN_DATA_PATH="/gpuhome/hbz5148/workspace/siyuan/ReMax/step4_xrlhf_finetuning/pythia-2.8b_unlearn.json"
RETAIN_DATA_PATH="/gpuhome/hbz5148/workspace/siyuan/ReMax/step4_xrlhf_finetuning/pythia-2.8b_retain.json"
EVAL_DATA_PATH="/gpuhome/hbz5148/workspace/siyuan/ReMax/dataset/Dahoas/full-hh-rlhf"
MODEL_NAME=~/workspace/siyuan/ReMax/step1_supervised_finetuning/output/pythia-2.8b/full-hh-rlhf
SEED=2023

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/pythia-2.8b/full-hh-rlhf
fi
mkdir -p $OUTPUT


(deepspeed --include localhost:$DEV --master_port $PORT \
main.py \
   --eval_data_path $EVAL_DATA_PATH \
   --unlearn_data_path $UNLEARN_DATA_PATH\
   --retain_data_path $RETAIN_DATA_PATH\
   --data_output_path "/tmp/data_files/opt" \
   --data_split 2,4,4 \
   --model_name_or_path $MODEL_NAME \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0. \
   --num_train_epochs 6  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed $SEED \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --print_loss \
   --deepspeed) 2>&1 | tee "$OUTPUT/training.log"