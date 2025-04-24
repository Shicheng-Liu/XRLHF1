#!/bin/bash

# trained by opt-1.3b reward model


CUDA_VISIBLE_DEVICES=0 python test.py \
    --model_name_or_path_baseline facebook/opt-1.3b \
    --model_name_or_path_xrlhf output/opt-1.3b/full-hh-rlhf/actor \
    --data_path /gpuhome/hbz5148/workspace/siyuan/ReMax/dataset/Dahoas/full-hh-rlhf/test.json \
    --batch_size 8 
    
