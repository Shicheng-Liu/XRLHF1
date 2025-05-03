#!/bin/bash

# trained by opt-1.3b reward model


CUDA_VISIBLE_DEVICES=6 python test_unsatisfactory.py \
    --model_name opt-1.3b \
    --model_name_or_path_baseline facebook/opt-1.3b \
    --model_name_or_path_xrlhf output/opt-1.3b/full-hh-rlhf \
    --data_path /gpuhome/hbz5148/workspace/siyuan/ReMax/step3_rlhf_finetuning/opt-1.3b_unsatisfactory.json \
    --batch_size 8 
    
