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

# DEV=1,2
# PORT=1237
# EVAL_DATA_PATH="/gpuhome/hbz5148/workspace/siyuan/ReMax/dataset/Dahoas/full-hh-rlhf"
# UNLEARN_DATA_PATH="/gpuhome/hbz5148/workspace/siyuan/ReMax/step4_xrlhf_finetuning/opt-1.3b_unlearn.json"
# RETAIN_DATA_PATH="/gpuhome/hbz5148/workspace/siyuan/ReMax/step4_xrlhf_finetuning/opt-1.3b_retain.json"
# ACTOR_MODEL_PATH=~/workspace/siyuan/ReMax/step1_supervised_finetuning/output/opt-1.3b/full-hh-rlhf
# OUTPUT=$1
# ZERO_STAGE=$2
# if [ "$OUTPUT" == "" ]; then
#     OUTPUT=./output/opt-1.3b/full-hh-rlhf
# fi
# if [ "$ZERO_STAGE" == "" ]; then
#     ZERO_STAGE=0
# fi
# mkdir -p $OUTPUT

# (deepspeed --include localhost:$DEV --master_port $PORT \
# main_dpo.py --actor_model_path $ACTOR_MODEL_PATH \
#    --unlearn_data_path $UNLEARN_DATA_PATH \
#    --retain_data_path $RETAIN_DATA_PATH \
#    --eval_data_path $EVAL_DATA_PATH \
#    --weight_decay 0.1 --dropout 0.0 --gradient_accumulation_steps 4 --zero_stage $ZERO_STAGE \
#    --enable_tensorboard \
#    --tensorboard_path $OUTPUT \
#    --deepspeed --output_dir $OUTPUT) 2>&1 | tee $OUTPUT/training.log


# init_model_name=$1

# dataset_name=hh/rw
# dataset_path=$3
# # options: exo-pref/exo-rw


# tb_path=tb_logs

# dataset_abbr=$( echo $dataset_name | cut -d'/' -f1 )

# general
dev=1,2
port=1484
init_model_path=~/workspace/siyuan/ReMax/step1_supervised_finetuning/output/opt-1.3b/full-hh-rlhf
beta_r=0.5
beta_pi=0.25
num_contrastive=2
temp=0.8
max_iter_step=700
save_step_interval=-1
num_save_checkpoint=20
loss_type="dpo"
dataset_name="full-hh-rlhf"
dataset_path="/gpuhome/hbz5148/workspace/siyuan/ReMax/dataset/Dahoas/full-hh-rlhf-step4"
data_output_path="data_output"
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output/opt-1.3b/full-hh-rlhf
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

train_bsz=8
eval_bsz=8
max_len=512
max_gen_len=200
lr=1e-6
wm_steps=0
eps=1
grad_accum=4
wd=0
ZERO_STAGE=2
exp_name="opt-1.3b"
# training commands ==================================


deepspeed --include localhost:$dev --master_port $port \
main_dpo.py \
   --model_name_or_path $init_model_path \
   --ref_name_or_path $init_model_path \
   --beta_r $beta_r \
   --beta_pi $beta_pi \
   --num_contrastive $num_contrastive \
   --temp $temp \
   --max_iter_step $max_iter_step \
   --save_step_interval $save_step_interval \
   --num_save_checkpoint $num_save_checkpoint \
   --loss_type $loss_type \
   --data_name_path $dataset_name:$dataset_path \
   --data_output_path $data_output_path \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --tensorboard_name_path $exp_name:$OUTPUT \
   --per_device_train_batch_size $train_bsz \
   --per_device_eval_batch_size $eval_bsz \
   --max_seq_len $max_len \
   --max_gen_len $max_gen_len \
   --learning_rate $lr \
   --num_warmup_steps $wm_steps \
   --num_train_epochs $eps \
   --gradient_accumulation_steps $grad_accum \
   --weight_decay $wd \
   --gradient_checkpointing \
   --print_loss \
   --zero_stage $ZERO_STAGE \
   --deepspeed 2>&1 | tee -a $OUTPUT/training.log