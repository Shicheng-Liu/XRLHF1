#!/bin/bash

set -x
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=20
export TOKENIZERS_PARALLELISM=False

DEV=0,2,4
PORT=1237
EVAL_DATA_PATH="/gpuhome/hbz5148/workspace/siyuan/ReMax/dataset/Dahoas/full-hh-rlhf"
UNLEARN_DATA_PATH="/gpuhome/hbz5148/workspace/siyuan/ReMax/step4_xrlhf_finetuning/opt-1.3b_unlearn.json"
RETAIN_DATA_PATH="/gpuhome/hbz5148/workspace/siyuan/ReMax/step4_xrlhf_finetuning/opt-1.3b_retain.json"
ACTOR_MODEL_PATH=~/workspace/siyuan/ReMax/step3_rlhf_finetuning/output/opt-1.3b/full-hh-rlhf/actor
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/opt-1.3b/full-hh-rlhf
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

(deepspeed --include localhost:$DEV --master_port $PORT \
main.py --actor_model_path $ACTOR_MODEL_PATH \
   --unlearn_data_path $UNLEARN_DATA_PATH \
   --retain_data_path $RETAIN_DATA_PATH \
   --eval_data_path $EVAL_DATA_PATH \
   --weight_decay 0.1 --dropout 0.0 --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT) 2>&1 | tee $OUTPUT/training.log
