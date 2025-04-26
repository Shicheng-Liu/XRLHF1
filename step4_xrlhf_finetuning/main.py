#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import time
import sys
import json
from copy import deepcopy
import hashlib
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from utils.data.data_utils import create_prompt_dataset, PromptDataset, get_raw_dataset
from utils.utils import (
    print_rank_0,
    to_device,
    save_hf_format,
    set_random_seed,
    save_code,
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    save_zero_three_model,
    load_hf_tokenizer,
)
from utils.ds_utils import get_train_ds_config
from utils.module.lora import (
    convert_linear_layer_to_lora,
    convert_lora_to_linear_layer,
    only_optimize_lora_parameters,
    make_model_gradient_checkpointing_compatible,
)
from utils.model.model_utils import create_hf_model
from utils.perf import print_throughput
from utils.gpu_utils import print_machine_info

IGNORE_INDEX = -100

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument('--unlearn_data_path',
                        nargs="*",
                        help='Path to the unlearn set',
                        required=True,
                        )
    parser.add_argument('--retain_data_path',
                        nargs="*",
                        help='Path to the retain set',
                        required=True,
                        )
    parser.add_argument('--eval_data_path',
                        nargs="*",
                        help='Path to the eval set',
                        required=True,
                        )
    parser.add_argument(
        "--data_split",
        type=str,
        default="2,4,4",
        help="Comma-separated list of proportions for training"
        "phase 1, 2, and 3 data. For example the split `6,2,2`"
        "will use 60%% of data for phase 1, 20%% for phase 2"
        "and 20%% for phase 3.",
    )
    parser.add_argument(
        "--sft_only_data_path",
        nargs="*",
        default=[],
        help="Path to the dataset for only using in SFT phase.",
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="/tmp/data_files/",
        help="Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)",
    )
    parser.add_argument(
        "--unlearn_data_output_path",
        type=str,
        default="/tmp/data_files/unlearn/",
        help="Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)",
    )
    parser.add_argument(
        "--retain_data_output_path",
        type=str,
        default="/tmp/data_files/retain/",
        help="Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=2023, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--data_seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--disable_dropout",
        action="store_true",
        help="Disable the dropout of the model.",
    )
    # deepspeed features
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument("--bf16", action="store_true", help="Enable bf16.")
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )
    ## LoRA for efficient training setting
    parser.add_argument(
        "--lora_dim",
        type=int,
        default=0,
        help="If > 0, use LoRA for efficient training.",
    )
    parser.add_argument(
        "--lora_module_name",
        type=str,
        default="decoder.layers.",
        help="The scope of LoRA.",
    )
    parser.add_argument(
        "--only_optimize_lora",
        action="store_true",
        help="Only optimize the LoRA parameters.",
    )
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial LoRA learning rate (after the potential warmup period) to use.",
    )
    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="step1_tensorboard")
    ## Print loss
    parser.add_argument(
        "--print_loss", action="store_true", help="Prints loss at each step."
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def get_dataset(local_rank,
        dataset_name,
        output_path,
        train_phase,
        seed,
        tokenizer,
        end_of_conversation_token,
        max_seq_len):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    for i, tmp_data in enumerate(train_dataset):
        prompt = raw_dataset.get_prompt(tmp_data)
        chosen_sentence = raw_dataset.get_prompt_and_chosen(
            tmp_data
        )  # the accept response
        # rejected_sentence = raw_dataset.get_prompt_and_rejected(
        #     tmp_data
        # )  
        if chosen_sentence is not None: # and rejected_sentence is not None:
            prompt_token = tokenizer(
                prompt,
                max_length=max_seq_len,
                padding="do_not_pad",
                truncation=True,
                return_tensors="pt",
            )
            prompt_length = len(prompt_token["input_ids"].flatten())

            chosen_sentence += end_of_conversation_token
            # rejected_sentence += end_of_conversation_token
            chosen_token = tokenizer(
                chosen_sentence,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # rejected_token = tokenizer(
            #     rejected_sentence,
            #     max_length=max_seq_len,
            #     padding="max_length",
            #     truncation=True,
            #     return_tensors="pt",
            # )
            chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(0)
            chosen_token["attention_mask"] = chosen_token["attention_mask"].squeeze(0)
            # rejected_token["input_ids"] = rejected_token["input_ids"].squeeze(0)
            # rejected_token["attention_mask"] = rejected_token["attention_mask"].squeeze(0)

            # do not calculate loss for prompts labels
            chosen_labels = deepcopy(chosen_token["input_ids"])
            # rejected_labels = deepcopy(rejected_token["input_ids"])
            if prompt_length == max_seq_len:
                # skip inputs that are all of prompts
                continue
            else:
                chosen_labels[:prompt_length] = IGNORE_INDEX
                # rejected_labels[:prompt_length] = IGNORE_INDEX

            # our implementation: pad tokens are not used for loss
            assert tokenizer.pad_token == tokenizer.eos_token
            chosen_padding_begin_ids = (
                (
                    chosen_token["input_ids"][prompt_length:]
                    == tokenizer.pad_token_id
                )
                .nonzero()
                .flatten()
            )
            # rejected_padding_begin_ids = (
            #     (
            #         rejected_token["input_ids"][prompt_length:]
            #         == tokenizer.pad_token_id
            #     )
            #     .nonzero()
            #     .flatten()
            # )
            if len(chosen_padding_begin_ids) > 1:
                # we use padding_begin_ids[1] because of the right-side shifting in calculating loss
                chosen_padding_begin_id = chosen_padding_begin_ids[1].item() + prompt_length
                chosen_labels[chosen_padding_begin_id:] = IGNORE_INDEX
            # if len(rejected_padding_begin_ids) > 1:
            #     # we use padding_begin_ids[1] because of the right-side shifting in calculating loss
            #     rejected_padding_begin_id = rejected_padding_begin_ids[1].item() + prompt_length
            #     rejected_labels[rejected_padding_begin_id:] = IGNORE_INDEX
            chosen_token["labels"] = chosen_labels
            chosen_dataset.append(chosen_token)
            # rejected_token["labels"] = rejected_labels
            # reject_dataset.append(rejected_token)
            #print(f"Keys in chosen_token[{i}]:", chosen_token.keys())
    return PromptDataset(
        prompt_dataset,
        chosen_dataset,
        reject_dataset,
        tokenizer.pad_token_id,
        train_phase,
    )

    
def get_prompt_dataset(local_rank,
    data_path,
    output_path,
    phase,
    seed,
    tokenizer,
    max_seq_len,
    end_of_conversation_token="<|endoftext|>",
    reload=True):
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_phase4_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(
        fname.encode()
    ).hexdigest()  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) 
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)
    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        train_dataset = get_dataset(
            local_rank,
            data_path[0],
            output_path,
            phase,
            seed,
            tokenizer,
            end_of_conversation_token,
            max_seq_len,
        )
        torch.save(train_dataset, train_fname)
    torch.distributed.barrier()
    return torch.load(train_fname,weights_only=False)

def save_model(
    model,
    tokenizer,
    args,
):
    if args.output_dir is not None:
        print_rank_0("saving the final model ...", args.global_rank)
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(
                model, args.global_rank, args.output_dir, zero_stage=args.zero_stage
            )


def main():
    args = parse_args()
    args.tensorboard_path = args.output_dir

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(
        offload=args.offload,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        bf16=args.bf16,
        tb_path=args.tensorboard_path,
        tb_name="",
    )
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()
    if args.global_rank == 0:
        with open(
            os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8"
        ) as f:
            for key, value in args.__dict__.items():
                json.dump({key: value}, f, ensure_ascii=False)
                f.write("\n")
        #save_code(args.output_dir)

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        ds_config,
        disable_dropout=args.disable_dropout,
    )

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(
            model, args.lora_module_name, args.lora_dim
        )
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)

    # Prepare the data
    eval_phase = 1
    unlearn_phase = 4
    retain_phase = 5
    _, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.eval_data_path,
        args.data_split,
        args.data_output_path,
        eval_phase,
        args.data_seed,
        tokenizer,
        args.max_seq_len,
        end_of_conversation_token=tokenizer.eos_token,
        sft_only_data_path=args.sft_only_data_path,
        reload=True,
    )
    unlearn_dataset = get_prompt_dataset(
        args.local_rank, args.unlearn_data_path,
        args.unlearn_data_output_path,
        unlearn_phase,
        args.seed,
        tokenizer,
        args.max_seq_len)
    retain_dataset = get_prompt_dataset(
        args.local_rank, args.retain_data_path,
        args.retain_data_output_path,
        retain_phase,
        args.seed,
        tokenizer,
        args.max_seq_len)
    # DataLoaders creation:
    if args.local_rank == -1:
        #train_sampler = RandomSampler(train_dataset)
        unlearn_sampler = RandomSampler(unlearn_dataset)
        retain_sampler = RandomSampler(retain_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        #train_sampler = DistributedSampler(train_dataset)
        unlearn_sampler = DistributedSampler(unlearn_dataset)
        retain_sampler = DistributedSampler(retain_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    unlearn_dataloader = DataLoader(unlearn_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=unlearn_sampler,
                                 batch_size=args.per_device_train_batch_size)
    retain_dataloader = DataLoader(retain_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=retain_sampler,
                                 batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()

            # if step == 99:
            #     break
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate
    )

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95)
    )

    num_update_steps_per_epoch = math.ceil(
        len(retain_dataloader) / args.gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    print_machine_info(args.global_rank)
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {1}/{args.num_train_epochs} *****",
        args.global_rank,
    )
    perplexity = evaluation(model, eval_dataloader)
    print_rank_0(f"ppl: {perplexity}", args.global_rank)
    if model.monitor.enabled and model.global_rank == 0:
        summary_events = [("Test/ppl", perplexity, model.global_samples)]
        model.monitor.write_events(summary_events)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(retain_dataloader)}",
            args.global_rank,
        )
        model.train()
        unlearn_iter = iter(unlearn_dataloader)
        for step, retain_batch in enumerate(retain_dataloader):
            start = time.time()
            model.zero_grad()

            retain_batch = to_device(retain_batch, device)

            try:
                unlearn_batch = next(unlearn_iter)
            except StopIteration:
                unlearn_iter = iter(unlearn_dataloader)
                unlearn_batch = next(unlearn_iter)
            unlearn_batch = to_device(unlearn_batch, device)
            # Combine inputs BEFORE the forward pass
            merged_batch = {
                key: torch.cat([retain_batch[key], unlearn_batch[key]], dim=0)
                for key in retain_batch
            }
            merged_batch = to_device(merged_batch, device)
            outputs = model(**merged_batch, use_cache=False)
            final_loss = outputs.loss
            # logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            # labels = merged_batch['labels']  # (batch_size, seq_len)

            # # 4. Split
            # batch_size = retain_batch['input_ids'].size(0)
            # retain_logits = logits[:batch_size]
            # unlearn_logits = logits[batch_size:]
            # retain_labels = labels[:batch_size]
            # unlearn_labels = labels[batch_size:]

            # # 5. Compute losses manually
            # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')  # or your loss config

            # # Reshape logits and labels if needed
            # retain_loss = loss_fct(retain_logits.view(-1, retain_logits.size(-1)), retain_labels.view(-1))
            # unlearn_loss = loss_fct(unlearn_logits.view(-1, unlearn_logits.size(-1)), unlearn_labels.view(-1))

            

            # retain_outputs = model(**retain_batch, use_cache=False)
            # retain_loss = retain_outputs.loss
            # unlearn_outputs = model(**unlearn_batch, use_cache=False)
            # unlearn_loss = unlearn_outputs.loss
            # final_loss = retain_loss + unlearn_loss
            if args.print_loss:
                print(
                    f"Epoch: {epoch + 1}, Step: {step}, Rank: {torch.distributed.get_rank()}, retain_loss = {final_loss}"
                )
            model.backward(final_loss)
            model.step()
            end = time.time()
            if torch.distributed.get_rank() == 0:
                print_throughput(model.module, args, end - start, args.global_rank)

            if (step + 1) % int(len(retain_dataloader) // 10) == 0:
                # Evaluate perplexity on the validation set.
                print_rank_0(
                    f"***** Evaluating perplexity, Epoch {epoch + 1}/{args.num_train_epochs} Step: {step} *****",
                    args.global_rank,
                )
                perplexity = evaluation(model, eval_dataloader)
                print_rank_0(f"ppl: {perplexity}", args.global_rank)
                print_machine_info(args.global_rank)
                if model.monitor.enabled and model.global_rank == 0:
                    summary_events = [("Test/ppl", perplexity, model.global_samples)]
                    model.monitor.write_events(summary_events)
                model.train()

                save_model(model, tokenizer, args)
        model.tput_timer.update_epoch_count()

    save_model(model, tokenizer, args)


if __name__ == "__main__":
    main()
