# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch

from transformers import (
    AutoModelForCausalLM, )

from dschat.utils.model.model_utils import create_hf_model, create_critic_model
from dschat.utils.utils import to_device, load_hf_tokenizer
from deepspeed import get_accelerator

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
        "--model_name_or_path_reward",
        type=str,
        help="Path to reward model",
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
                                dropout=0.)

    return model, tokenizer

def prepare_singlesample(prompt,
                         good_ans,
                         tokenizer,
                         max_seq_len=512,
                         end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans[0] + end_of_conversation_token
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = chosen_token["input_ids"]
    batch["attention_mask"] = chosen_token["attention_mask"]

    return batch


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


def prompt_eval(args, model_baseline, model_fintuned, model_rlhf, tokenizer, reward_model, device, prompts):
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Ensure the pad_token is set, especially if it's the same as eos_token
        # tokenizer.pad_token = tokenizer.eos_token
        # model_baseline.config.pad_token_id = tokenizer.pad_token_id
        # model_fintuned.config.pad_token_id = tokenizer.pad_token_id

        # Manually set the attention mask if it's not already set
        # if 'attention_mask' not in inputs:
        #     inputs['attention_mask'] = (inputs['input_ids'] != tokenizer.pad_token_id).to(torch.long)
        
        print("==========Baseline: Greedy=========")
        r_base = generate(model_baseline,
                          tokenizer,
                          inputs,
                          num_beams=1,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)
        print_utils(r_base)
        base_batch = prepare_singlesample(prompt, r_base, tokenizer, max_seq_len=512, end_of_conversation_token=args.end_of_conversation_token)
        base_batch = to_device(base_batch, device)
        reward_model.eval()
        # Run inference
        with torch.no_grad():
            base_outputs = reward_model.forward_value(
                **base_batch, prompt_length=max(2, args.num_padding_at_beginning)
            )
        print("baseline answer score: ", base_outputs["chosen_end_scores"].item())

        print("==========finetune: Greedy=========")
        r_finetune_g = generate(model_fintuned,
                                tokenizer,
                                inputs,
                                num_beams=1,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        print_utils(r_finetune_g)
        finetune_batch = prepare_singlesample(prompt, r_finetune_g, tokenizer, max_seq_len=512, end_of_conversation_token=args.end_of_conversation_token)
        finetune_batch = to_device(finetune_batch, device)
        
        # Run inference
        with torch.no_grad():
            finetune_outputs = reward_model.forward_value(
                **finetune_batch, prompt_length=max(2, args.num_padding_at_beginning)
            )
        print("finetune answer score: ", finetune_outputs["chosen_end_scores"].item())

        print("==========rlhf: Greedy=========")
        r_rlhf_g = generate(model_rlhf,
                                tokenizer,
                                inputs,
                                num_beams=1,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        print_utils(r_rlhf_g)
        rlhf_batch = prepare_singlesample(prompt, r_rlhf_g, tokenizer, max_seq_len=512, end_of_conversation_token=args.end_of_conversation_token)
        rlhf_batch = to_device(rlhf_batch, device)
        
        # Run inference
        with torch.no_grad():
            rlhf_outputs = reward_model.forward_value(
                **rlhf_batch, prompt_length=max(2, args.num_padding_at_beginning)
            )
        print("rlhf answer score: ", rlhf_outputs["chosen_end_scores"].item())
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
        print("====================prompt end=============================")
        print()
        print()


def main():
    args = parse_args()

    device = torch.device(get_accelerator().device_name(0))

    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.model_name_or_path_baseline,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    model_baseline = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_baseline,
                                     tokenizer, None)
    model_fintuned = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_finetune,
                                     tokenizer, None)

    model_rlhf = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_rlhf,
                                     tokenizer, None)
    
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None

    reward_model, _ = load_stuff(args.model_name_or_path_reward,
                                     args.num_padding_at_beginning,
                                     additional_special_tokens)

    model_baseline.to(device)
    model_fintuned.to(device)
    model_rlhf.to(device)
    reward_model.to(device)
    

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    if args.language == "English":
        prompts = [
            "\n\nHuman: Please tell me about Microsoft in a few sentence?\n\nAssistant:",
            "\n\nHuman: Explain the moon landing to a 6 year old in a few sentences.\n\nAssistant:",
            "\n\nHuman: Write a short poem about a wise frog.\n\nAssistant:",
            "\n\nHuman: Who was president of the United States in 1955?\n\nAssistant:",
            "\n\nHuman: How does a telescope work?\n\nAssistant:",
            "\n\nHuman: Why do birds migrate south for the winter?\n\nAssistant:"
        ]
    elif args.language == "Chinese":
        prompts = [
            "Human: 请用几句话介绍一下微软? Assistant:",
            "Human: 用几句话向6岁的孩子解释登月。 Assistant:",
            "Human: 写一首关于一只聪明的青蛙的短诗。 Assistant:",
            "Human: 谁是1955年的美国总统? Assistant:", "Human: 望远镜是如何工作的? Assistant:",
            "Human: 鸟类为什么要南迁过冬? Assistant:"
        ]
    elif args.language == "Japanese":
        prompts = [
            "Human: マイクロソフトについて簡単に教えてください。 Assistant:",
            "Human: 6歳児に月面着陸を短い文で説明する。 Assistant:",
            "Human: 賢いカエルについて短い詩を書いてください。 Assistant:",
            "Human: 1955年のアメリカ合衆国大統領は誰? Assistant:",
            "Human: 望遠鏡はどのように機能しますか? Assistant:",
            "Human: 鳥が冬に南に移動するのはなぜですか? Assistant:"
        ]

    prompt_eval(args, model_baseline, model_fintuned, model_rlhf, tokenizer, reward_model, device,
                prompts)


if __name__ == "__main__":
    main()
