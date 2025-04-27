#!/bin/bash

# trained by opt-1.3b reward model

MODEL_NAME="PKU-Alignment/beaver-7b-v3.0-reward"
#MODEL_NAME=~/workspace/siyuan/ReMax/step2_reward_model_finetuning/output/opt-1.3b/full-hh-rlhf

CUDA_VISIBLE_DEVICES=1 python reward_eval.py \
    --data_path opt-1.3b_tldr_test_result.json \
    --model_name_or_path_reward $MODEL_NAME \
    --data_name tldr
    
