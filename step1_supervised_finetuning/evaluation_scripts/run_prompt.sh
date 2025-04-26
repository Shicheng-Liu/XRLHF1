#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model

CUDA_VISIBLE_DEVICES=0 python prompt_eval.py \
    --model_name_or_path_baseline EleutherAI/pythia-2.8b \
    --model_name_or_path_finetune output/pythia-2.8b/full-hh-rlhf
