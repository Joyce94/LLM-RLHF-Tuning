import argparse
import json
import os
import gc
import torch
import peft
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from trl import AutoModelForCausalLMWithValueHead
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument('--base_model_path', default=None,
                    type=str, help="Please specify a base_model")
parser.add_argument('--adpter_model', default=None,
                    type=str, help="Please specify LoRA models to be merged (ordered); use commas to separate multiple LoRA models.")
parser.add_argument('--output_dir', default=None, type=str)


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def save(state_dict, output_dir):
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


if __name__=='__main__':

    args = parser.parse_args()
    base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model_path
    )
    base_model.config.save_pretrained(args.output_dir)

    tokenizer = LlamaTokenizer.from_pretrained(args.base_model_path)
    tokenizer.save_pretrained(args.output_dir)

    base_model_sd = base_model.state_dict()
    # print(list(base_model_sd.keys()))
    adpter_model_sd = torch.load(os.path.join(args.adpter_model,'adapter_model.bin'),map_location='cpu')
    # print(list(adpter_model_sd.keys()))

    config = peft.LoraConfig.from_pretrained(args.adpter_model)
    scaling = config.lora_alpha / config.r
    fan_in_fan_out = config.fan_in_fan_out
    expected_keys = [k for k in adpter_model_sd if 'lora_A' in k]
    non_expected_keys = [k for k in adpter_model_sd if not 'lora_' in k]

    for k in non_expected_keys:
        print(f"merging {k}")
        original_k = k.replace('base_model.model.','')
        base_model_sd[original_k].copy_(adpter_model_sd[k])

    for k in expected_keys:
        print(f"merging {k}")
        original_key = k.replace('.lora_A','').replace('base_model.model.','')
        assert original_key in base_model_sd
        lora_a_key = k
        lora_b_key = k.replace('lora_A','lora_B')
        base_model_sd[original_key] += (
            transpose(adpter_model_sd[lora_b_key].float() @ adpter_model_sd[lora_a_key].float(),fan_in_fan_out) * scaling
        )

    print("Saving to Hugging Face format...")
    # AutoModelForCausalLMWithValueHead.save_pretrained(base_model_sd, args.output_dir) 
    torch.save(base_model_sd, os.path.join(args.output_dir, "pytorch_model.bin"))
