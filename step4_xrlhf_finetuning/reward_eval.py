# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from tqdm import tqdm
from utils.model.model_utils import create_critic_model
from utils.utils import to_device, load_hf_tokenizer
from deepspeed import get_accelerator
from datasets import load_dataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_reward",
        type=str,
        help="Path to reward model",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to test prompts",
        required=True,
    )
    parser.add_argument(
        "--new_data_path",
        type=str,
        help="Path to test prompts",
        required=True,
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


def load_stuff(model_name_or_path, num_padding_at_beginning,
               additional_special_tokens):

    tokenizer = load_hf_tokenizer(model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)
    tokenizer.pad_token = tokenizer.eos_token
    model = create_critic_model(model_name_or_path,
                                tokenizer,
                                None,
                                num_padding_at_beginning,
                                rlhf_training=True,
                                disable_dropout=True)

    return model, tokenizer

def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len=512,
                         end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans + end_of_conversation_token
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = chosen_token["input_ids"]
    batch["attention_mask"] = chosen_token["attention_mask"]

    return batch

def PKU_reward(prompt,response,reward_model,reward_tokenizer,device):
    input = prompt + response
    input_ids = reward_tokenizer(input,return_tensors='pt',truncation=True,max_length=2048)
    input_ids = to_device(input_ids,device)
    output = reward_model(**input_ids)
    return output.end_scores.item()

def opt_reward(prompt,response,reward_model,reward_tokenizer,device,end_of_conversation_token,num_padding_at_beginning):
    batch = prepare_singlesample(prompt, response, reward_tokenizer, max_seq_len=512, end_of_conversation_token=end_of_conversation_token)
    batch = to_device(batch, device)
    reward_model.eval()
        # Run inference
    with torch.no_grad():
        outputs = reward_model.forward_value(**batch, prompt_length=max(2, num_padding_at_beginning))
        return outputs["chosen_end_scores"].item()
    
def OpenAssistant_reward(prompt,response,reward_model,reward_tokenizer,device):
    input_ids = reward_tokenizer(prompt,response,return_tensors='pt')
    input_ids = to_device(input_ids,device)
    reward = reward_model(**input_ids).logits[0].cpu().detach()
    return reward
    
       
def get_reward(prompt,response,reward_model,reward_tokenizer,device,end_of_conversation_token,num_padding_at_beginning,reward_name):
    if "opt" in reward_name:
        reward = opt_reward(prompt,response,reward_model,reward_tokenizer,device,end_of_conversation_token,num_padding_at_beginning)
    if "PKU" in reward_name:
        reward = PKU_reward(prompt,response,reward_model,reward_tokenizer,device)
    if "OpenAssistant" in reward_name:
        reward = OpenAssistant_reward(prompt,response,reward_model,reward_tokenizer,device)
    return reward




def main():
    args = parse_args()

    device = torch.device(get_accelerator().device_name(0))

    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None

    if "opt" in args.model_name_or_path_reward:
        reward_model, reward_tokenizer = load_stuff(args.model_name_or_path_reward,
                                        args.num_padding_at_beginning,
                                        additional_special_tokens)
    elif "PKU-Alignment" in args.model_name_or_path_reward:
        from safe_rlhf.models import AutoModelForScore
        reward_model = AutoModelForScore.from_pretrained(args.model_name_or_path_reward, torch_dtype=torch.bfloat16, device_map='auto')
        reward_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path_reward)
    elif "OpenAssistant" in args.model_name_or_path_reward:
        reward_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path_reward)
        reward_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path_reward)
    else:
        #from huggingface_hub import login
        #login(token="")
        reward_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path_reward)
        reward_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path_reward, device_map="auto", torch_dtype="auto")
    reward_model.to(device)
    

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    ds = load_dataset("json", data_files=args.data_path)["train"]
    new_ds = load_dataset("json", data_files=args.new_data_path)["train"]
    prompts = ds["prompt"]  
    
    response_sft = ds["response_sft"]
    response_rlhf = ds["response_rlhf"]
    response_xrlhf = new_ds["response_xrlhf"]

    
    
    
    reward_finetune_list = []
    reward_rlhf_list = []
    reward_xrlhf_list = []
    win_rate_list_rlhf_sft = []
    win_rate_list_xrlhf_sft = []
    win_rate_list_xrlhf_rlhf = []
    
    for prompt, sft_response, rlhf_response, xrlhf_response in tqdm(zip(prompts, response_sft, response_rlhf, response_xrlhf),total=len(prompts),desc="Evaluation process"):
        
        # print('base_response',base_response)
        # print('sft_response',sft_response)
        # print('rlhf_response',rlhf_response)

        
        finetune_reward = get_reward(prompt,sft_response,reward_model,reward_tokenizer,device,args.end_of_conversation_token,args.num_padding_at_beginning,args.model_name_or_path_reward)
        rlhf_reward = get_reward(prompt,rlhf_response,reward_model,reward_tokenizer,device,args.end_of_conversation_token,args.num_padding_at_beginning,args.model_name_or_path_reward)
        xrlhf_reward = get_reward(prompt,xrlhf_response,reward_model,reward_tokenizer,device,args.end_of_conversation_token,args.num_padding_at_beginning,args.model_name_or_path_reward)

        
        reward_finetune_list.append(finetune_reward)
        reward_rlhf_list.append(rlhf_reward)
        reward_xrlhf_list.append(xrlhf_reward)
        if rlhf_reward >= finetune_reward:
            win_rate_list_rlhf_sft.append(1)
        else:
            win_rate_list_rlhf_sft.append(0)
        if xrlhf_reward >= finetune_reward:
            win_rate_list_xrlhf_sft.append(1)
        else:
            win_rate_list_xrlhf_sft.append(0)
        if xrlhf_reward >= rlhf_reward:
            win_rate_list_xrlhf_rlhf.append(1)
        else:
            win_rate_list_xrlhf_rlhf.append(0)
        # elif rlhf_reward < finetune_reward:
        #     win_rate_list.append(0)
        # elif rlhf_reward == finetune_reward and sign == 1:
        #     win_rate_list.append(1)
        #     sign = 0
        # else:
        #     win_rate_list.append(0)
        #     sign = 1

        

    
    print("reward for SFT model",np.mean(reward_finetune_list))
    print("reward for rlhf model",np.mean(reward_rlhf_list))
    print("reward for xrlhf model",np.mean(reward_xrlhf_list))
    print("RLHF_SFT win rate",1.0*sum(win_rate_list_rlhf_sft)/len(win_rate_list_rlhf_sft))
    print("XRLHF_SFT win rate",1.0*sum(win_rate_list_xrlhf_sft)/len(win_rate_list_xrlhf_sft))
    print("XRLHF_RLHF win rate",1.0*sum(win_rate_list_xrlhf_rlhf)/len(win_rate_list_xrlhf_rlhf))


if __name__ == "__main__":
    main()
