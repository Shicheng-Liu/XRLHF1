# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from utils.model.model_utils import create_hf_model
from utils.utils import to_device, load_hf_tokenizer
from deepspeed import get_accelerator
from datasets import load_dataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_rlhf",
        type=str,
        help="Path to rlhf model",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="model name",
        required=True,
    )
    parser.add_argument(
        "--data_name",
        type=str,
        help="data name",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to test prompts",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help='batch size',
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")

    args = parser.parse_args()

    return args

def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  attention_mask=inputs.attention_mask,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


def prompt_eval(args, model_baseline, model_fintuned, model_rlhf, tokenizer, device, prompts):

    base_response = []
    finetune_response = []
    rlhf_response = []
    for prompt in tqdm(prompts, desc="Generating responses"):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        prompt_length = len(prompt)
        # Ensure the pad_token is set, especially if it's the same as eos_token
        # tokenizer.pad_token = tokenizer.eos_token
        # model_baseline.config.pad_token_id = tokenizer.pad_token_id
        # model_fintuned.config.pad_token_id = tokenizer.pad_token_id

        # Manually set the attention mask if it's not already set
        # if 'attention_mask' not in inputs:
        #     inputs['attention_mask'] = (inputs['input_ids'] != tokenizer.pad_token_id).to(torch.long)
        
        #print("==========Baseline: Greedy=========")
        r_base = generate(model_baseline,
                          tokenizer,
                          inputs,
                          num_beams=1,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)
        base_response.append(r_base[0][prompt_length:])
        #print_utils(r_base)

        #print("==========finetune: Greedy=========")
        r_finetune_g = generate(model_fintuned,
                                tokenizer,
                                inputs,
                                num_beams=1,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        #print_utils(r_finetune_g)
        finetune_response.append(r_finetune_g[0][prompt_length:])
        

        #print("==========rlhf: Greedy=========")
        r_rlhf_g = generate(model_rlhf,
                                tokenizer,
                                inputs,
                                num_beams=1,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        #print_utils(r_rlhf_g)
        rlhf_response.append(r_rlhf_g[0][prompt_length:])
        

    test_results = []
    for p, b, s, r in zip(prompts, base_response, finetune_response, rlhf_response):
        test_results.append({
            "prompt": p,
            "response_base": b,
            "response_sft": s,
            "response_rlhf": r
        })
    with open(f"{args.model_name}_{args.data_name}_test_result.json","w") as f:
        json.dump(test_results,f,indent=4)
        # Note: we use the above simplest greedy search as the baseline. Users can also use other baseline methods,
        # such as beam search, multinomial sampling, and beam-search multinomial sampling.
        # We provide examples as below for users to try.

        # print("==========finetune: Multinomial sampling=========")
        # r_finetune_m = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=1,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_m)
        # print("==========finetune: Beam Search=========")
        # r_finetune_b = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_b)
        # print("==========finetune: Beam-search multinomial sampling=========")
        # r_finetune_s = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_s)
        # print("==========finetune: Diverse Beam Search=========")
        # r_finetune_d = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_beam_groups=args.num_beam_groups,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_d)
        # print("==========finetune: Constrastive Search=========")
        # r_finetune_c = generate_constrastive_search(model_fintuned, tokenizer, inputs,
        #                                             top_k=args.top_k,
        #                                             penalty_alpha=args.penalty_alpha,
        #                                             num_return_sequences=args.num_return_sequences,
        #                                             max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_c)
        #print("====================prompt end=============================")
        #print()
        #print()



def main():
    args = parse_args()

    device = torch.device(get_accelerator().device_name(0))

    args.end_of_conversation_token = "<|endoftext|>"
    #additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.model_name_or_path_baseline,
                                  fast_tokenizer=True)  #,add_special_tokens=additional_special_tokens
    model_baseline = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_baseline,
                                     tokenizer, None)
    model_fintuned = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_finetune,
                                     tokenizer, None)
    model_rlhf = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_rlhf,
                                     tokenizer, None)
    
    model_baseline.to(device)
    model_fintuned.to(device)
    model_rlhf.to(device)
    

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    ds = load_dataset("json", data_files=args.data_path)["train"]
    prompts = ds["prompt"][:500]
    print("enter prompt")
    prompt_eval(args, model_baseline, model_fintuned, model_rlhf, tokenizer, device, prompts)


if __name__ == "__main__":
    main()
