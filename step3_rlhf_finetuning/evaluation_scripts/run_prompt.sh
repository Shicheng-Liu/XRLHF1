#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model

CUDA_VISIBLE_DEVICES=4 python prompt_eval.py \
    --model_name_or_path_baseline facebook/opt-1.3b \
    --model_name_or_path_finetune ~/workspace/siyuan/ReMax/training/step1_supervised_finetuning/output/opt-1.3b/full-hh-rlhf \
    --model_name_or_path_rlhf output/opt-350m/full-hh-rlhf/actor \
    --model_name_or_path_reward ~/workspace/siyuan/ReMax/training/step2_reward_model_finetuning/output/opt-350m/full-hh-rlhf
