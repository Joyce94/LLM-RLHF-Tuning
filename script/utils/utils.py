import os,sys
import torch 
import torch.nn as nn 

PROMPT_TEMPLATE = dict(
    chinese_llama_alpaca=(
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        ),
    chinese_llama2_alpaca=(
        "[INST] <<SYS>>\n"
        "You are a helpful assistant. 你是一个乐于助人的助手。\n"
        "<</SYS>>\n\n{instruction} [/INST]"
    ),
    default=(
        "Human: {instruction}\nAssistant: "
    )
)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)
























