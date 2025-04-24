#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math
import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import hashlib
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator
from utils.data.data_utils import get_raw_dataset, PromptDataset

from utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config, get_eval_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss
from utils.perf import print_throughput


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--unlearn_data_path',
                        type=str,
                        help='Path to the unlearn set',
                        required=True,
                        )
    parser.add_argument('--retain_data_path',
                        type=str,
                        help='Path to the retain set',
                        required=True,
                        )
    parser.add_argument('--eval_data_path',
                        type=str,
                        help='Path to the eval set',
                        required=True,
                        )
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        '--eval_data_output_path',
        type=str,
        default='/tmp/data_files/eval/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--actor_model_path",
        type=str,
        help=
        "Path to rlhf actor",
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
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-1,
        help=
        "Temperature parameter, typically something in the range of 0.1 to 0.5."
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help=
        "conservativeness for loss, which assumes that preferences are noisy (flipped with probability label_smoothing)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step4_tensorboard")
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def get_dataset(local_rank,
        dataset_name,
        output_path,
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
        # tokenize the text
        chosen_sentence = raw_dataset.get_prompt_and_chosen(
            tmp_data
        )  # the accept response
        reject_sentence = raw_dataset.get_prompt_and_rejected(
            tmp_data
        )  # the accept response
        if chosen_sentence is not None and reject_sentence is not None:
            chosen_sentence += end_of_conversation_token  # the accept response
            reject_sentence += end_of_conversation_token
            chosen_token = tokenizer(
                chosen_sentence,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            reject_token = tokenizer(
                reject_sentence,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            chosen_token["input_ids"] = chosen_token["input_ids"]
            chosen_token["attention_mask"] = chosen_token["attention_mask"]
            chosen_dataset.append(chosen_token)

            reject_token["input_ids"] = reject_token["input_ids"]
            reject_token["attention_mask"] = reject_token["attention_mask"]
            reject_dataset.append(reject_token)
    return PromptDataset(
        prompt_dataset,
        chosen_dataset,
        reject_dataset,
        tokenizer.pad_token_id,
        train_phase=2,
    )

    
def get_prompt_dataset(local_rank,
    data_path,
    output_path,
    seed,
    tokenizer,
    max_seq_len,
    end_of_conversation_token="<|endoftext|>",
    reload=False):
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
            seed,
            tokenizer,
            end_of_conversation_token,
            max_seq_len,
        )
        torch.save(train_dataset, train_fname)
    torch.distributed.barrier()
    return torch.load(train_fname,weights_only=False)


def get_batch_logps(logits, input_ids, label_mask):
    labels = input_ids.clone() * label_mask
    assert logits.shape[:-1] == labels.shape, \
        "Logits (batch and sequence length dim) and labels must have the same shape."
    labels = labels[:, 1:]
    label_mask = label_mask[:, 1:]
    logits = logits[:, :-1, :]
    per_token_logps = torch.gather(logits.log_softmax(-1),
                                   dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * label_mask).sum(-1)

def compute_loss(model, ref_model, tokenizer, batch, args):
    batch_size = batch['input_ids'].shape[0] // 2
    chosen_input_ids = batch['input_ids'][:batch_size]
    rejected_input_ids = batch['input_ids'][batch_size:]
    label_mask = (batch['input_ids'] != tokenizer.pad_token_id).int()

    for i in range(batch_size):
        divergence_ind = (chosen_input_ids[i] != rejected_input_ids[i]).nonzero().squeeze(-1)
        divergence_ind = divergence_ind[0] if len(divergence_ind) > 0 else 0
        label_mask[i][:divergence_ind] = 0
        label_mask[i + batch_size][:divergence_ind] = 0

    outputs = model(**batch, use_cache=False)
    with torch.no_grad():
        ref_outputs = ref_model(**batch)

    logps = get_batch_logps(outputs.logits, batch['input_ids'], label_mask)
    ref_logps = get_batch_logps(ref_outputs.logits, batch['input_ids'], label_mask)

    chosen_logps = logps[:batch_size]
    rejected_logps = logps[batch_size:]
    ref_chosen_logps = ref_logps[:batch_size]
    ref_rejected_logps = ref_logps[batch_size:]

    logits = args.beta * ((chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps))
    loss = (- torch.nn.functional.logsigmoid(logits) * (1 - args.label_smoothing) -
            torch.nn.functional.logsigmoid(-logits) * args.label_smoothing).mean(0)

    return loss

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    bf16=True,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step4_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.actor_model_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    model = create_hf_model(AutoModelForCausalLM,
                            args.actor_model_path,
                            tokenizer,
                            ds_config,
                            disable_dropout=True)

    # DS Config for ref model
    ref_zero_stage = args.zero_stage
    if ref_zero_stage != 3:
        # If it is ZeRO-3 then we use it for everything, otherwise assume we have enough memory for ref model
        ref_zero_stage = 0
    ref_ds_config = get_eval_ds_config(args.offload_reference_model,
                                       args.dtype, ref_zero_stage)
    ref_ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ref_ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    ref_ds_eval_config = get_eval_ds_config(offload=False,
                                            dtype=args.dtype,
                                            stage=ref_zero_stage)
    ref_ds_eval_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ref_ds_eval_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    ref_model = create_hf_model(AutoModelForCausalLM,
                                args.actor_model_path,
                                tokenizer,
                                ref_ds_eval_config,
                                disable_dropout=True)
    # End of DS config for ref model

    if args.compute_fp32_loss:
        print_rank_0(
            f"Using model {model.__class__.__name__} with loss in fp32",
            args.global_rank)
        causal_lm_model_to_fp32_loss(model)

    # Copied from ../step2_reward_model_finetuning/main.py.
    # Model bigscience/bloom-560m has large variance at ln_f.weight parameter
    # This makes bf16 finetuning hard.
    # In general, since we are replacing the model head, it makes sense to reset
    # the LN that precedes it.
    force_optimize_params = []
    if "bigscience/bloom-" in args.actor_model_path:
        zero_init_enabled = (args.zero_stage == 3)
        params = [
            model.rwtranrsformer.ln_f.weight, model.rwtranrsformer.ln_f.bias
        ]
        with deepspeed.zero.GatheredParameters(params,
                                               modifier_rank=0,
                                               enabled=zero_init_enabled):
            if deepspeed.comm.get_rank() == 0 or not zero_init_enabled:
                torch.nn.init.ones_(model.rwtransformer.ln_f.weight)
                torch.nn.init.zeros_(model.rwtransformer.ln_f.bias)
        force_optimize_params.extend(
            ['rwtransformer.ln_f.weight', 'rwtransformer.ln_f.bias'])

    if args.lora_dim > 0:
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)

    # Prepare the data
    train_phase = 2
    _, eval_dataset = create_prompt_dataset(
        args.local_rank, args.eval_data_path, args.data_split,
        args.eval_data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len)
    unlearn_dataset = get_prompt_dataset(
        args.local_rank, args.unlearn_data_path,
        args.output_path,
        args.seed,
        tokenizer,
        args.max_seq_len)
    retain_dataset = get_prompt_dataset(
        args.local_rank, args.retain_data_path,
        args.output_path,
        args.seed,
        tokenizer,
        args.max_seq_len)
    # DataLoaders creation:
    data_collator = DataCollatorReward()
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
    # train_dataloader = DataLoader(train_dataset,
    #                               collate_fn=data_collator,
    #                               sampler=train_sampler,
    #                               batch_size=args.per_device_train_batch_size)
    unlearn_dataloader = DataLoader(unlearn_dataset,
                                 collate_fn=data_collator,
                                 sampler=unlearn_sampler,
                                 batch_size=args.per_device_train_batch_size)
    retain_dataloader = DataLoader(retain_dataset,
                                 collate_fn=data_collator,
                                 sampler=retain_sampler,
                                 batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation(model, ref_model, tokenizer, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            batch_size = batch['input_ids'].shape[0] // 2
            chosen_input_ids = batch['input_ids'][:batch_size]
            rejected_input_ids = batch['input_ids'][batch_size:]
            label_mask = (batch['input_ids'] != tokenizer.pad_token_id).int()
            for i in range(batch_size):
                divergence_ind = (chosen_input_ids[i] !=
                                  rejected_input_ids[i]).nonzero().squeeze(-1)
                if len(divergence_ind) > 0:
                    divergence_ind = divergence_ind[0]
                else:
                    divergence_ind = 0
                label_mask[i][:divergence_ind] = 0
                label_mask[i + batch_size][:divergence_ind] = 0
            with torch.no_grad():
                outputs = model(**batch)
                ref_outputs = ref_model(**batch)

            logps = get_batch_logps(outputs.logits, batch['input_ids'],
                                    label_mask)
            ref_logps = get_batch_logps(ref_outputs.logits, batch['input_ids'],
                                        label_mask)

            chosen_logps = logps[:batch_size]
            rejected_logps = logps[batch_size:]
            ref_chosen_logps = ref_logps[:batch_size]
            ref_rejected_logps = ref_logps[batch_size:]

            logits = args.beta * ((chosen_logps - ref_chosen_logps) -
                                  (rejected_logps - ref_rejected_logps))
            loss = (- torch.nn.functional.logsigmoid(logits) * (1 - args.label_smoothing) - \
                    torch.nn.functional.logsigmoid(-logits) * args.label_smoothing).mean(0)
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except:
            pass
        chosen_rewards = args.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = args.beta * (rejected_logps -
                                        ref_rejected_logps).detach()
        return chosen_rewards.mean().item(), rejected_rewards.mean().item(
        ), losses.item()

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(retain_dataloader) / args.gradient_accumulation_steps)
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
        dist_init_required=True)
    ref_model, *_ = deepspeed.initialize(model=ref_model, config=ref_ds_config)
    ref_model.eval()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(
        f"***** Evaluating rewards, Epoch {1}/{args.num_train_epochs} *****",
        args.global_rank)
    chosen_rewards, rejected_rewards, eval_loss = evaluation(
        model, ref_model, tokenizer, eval_dataloader)
    print_rank_0(
        f"chosen: {chosen_rewards}, rejected: {rejected_rewards}, loss: {eval_loss}",
        args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(retain_dataloader)}",
            args.global_rank)
        model.train()
        import time
        unlearn_iter = iter(unlearn_dataloader)
        for step, retain_batch in enumerate(retain_dataloader):
            start = time.time()
            try:
                unlearn_batch = next(unlearn_iter)
            except StopIteration:
                # when unlearn is exhausted, recreate a new iterator, shuffling again
                unlearn_iter = iter(unlearn_dataloader)
                unlearn_batch = next(unlearn_iter)
            retain_batch = to_device(retain_batch, device)
            unlearn_batch = to_device(unlearn_batch, device)
            
            retain_loss = compute_loss(model, ref_model, tokenizer, retain_batch, args)
            unlearn_loss = compute_loss(model, ref_model, tokenizer, unlearn_batch, args)
            total_loss = retain_loss - unlearn_loss
            if args.print_loss:
                print(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {total_loss}"
                )
            model.backward(total_loss)
            model.step()
            end = time.time()
            if torch.distributed.get_rank() == 0:
                print_throughput(model.model, args, end - start,
                                 args.global_rank)

        # Evaluate rewards on the validation set.
        print_rank_0(
            f"***** Evaluating rewards, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        chosen_rewards, rejected_rewards, eval_loss = evaluation(
            model, ref_model, tokenizer, eval_dataloader)
        print_rank_0(
            f"chosen: {chosen_rewards}, rejected: {rejected_rewards}, loss: {eval_loss}",
            args.global_rank)
        model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        model = convert_lora_to_linear_layer(model)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()

