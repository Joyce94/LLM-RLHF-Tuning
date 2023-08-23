import os,re,sys
import torch 
import torch.nn as nn 
from transformers import AutoConfig,AutoTokenizer,LlamaForCausalLM,LlamaTokenizer,Trainer,AutoModelForCausalLM,get_scheduler,BitsAndBytesConfig,OPTForCausalLM,LlamaModel
import logging 
from peft import LoraConfig,PeftModel,TaskType,get_peft_model
from trl import AutoModelForCausalLMWithValueHead
from torch.optim import AdamW
from peft.tuners.lora import LoraLayer 
import deepspeed 

MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaTokenizer, LlamaForCausalLM),
    "opt": (AutoConfig, AutoTokenizer, OPTForCausalLM),
    "auto": (AutoConfig, AutoTokenizer, AutoModelForCausalLM),
}



def print_trainable_params(model: torch.nn.Module) -> None:
    # Adopted from https://github.com/LLaMA-Efficient-Tuning-main/src/utils/other.py
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
                trainable_params, all_param, 100 * trainable_params / all_param))


class PPOEngine():
    def __init__(self, model_args, training_args):
        self.model_args = model_args
        self.training_args = training_args

        self.config_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype),
            "low_cpu_mem_usage": True,
        }
        if self.model_args.load_in_4bit:
            self.config_kwargs["load_in_4bit"] = True
            self.config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        
        if model_args.sft_model_path is None:
            raise ValueError(
                f"you need to provide sft model path, but now model path is {model_args.sft_model_path}."
            )
        else:
            ## sft model使用actor model在disable adapter结果
            self.actor_model = self._init_actor()
            self.sft_model = None 

        if model_args.reward_model_path is None and model_args.reward_lora_path is None:
            raise ValueError(
                f"you need to provide reward model path, but now model path is {model_args.reward_model_path}."
            )
        else:
            ## reward_model使用critic_model在disable adapter结果
            self.critic_model = self._init_critic()
            self.reward_model = None


    def _create_sft_model(self):
        config_class, tokenizer_class, model_class = MODEL_CLASSES[self.model_args.model_type]
        model = model_class.from_pretrained(
            self.model_args.sft_model_path,
            from_tf=bool(".ckpt" in self.model_args.sft_model_path),
            **self.config_kwargs
        )
        
        model.requires_grad_(False)
        print('*********************sft*******************')
        print_trainable_params(model)

        return model

    def _create_rm_model(self):
        
        if self.model_args.reward_model_path is not None:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model_args.reward_model_path,
                from_tf=bool(".ckpt" in self.model_args.reward_model_path),
                **self.config_kwargs
            )
        elif self.model_args.reward_lora_path is not None:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model_args.sft_model_path,
                from_tf=bool(".ckpt" in self.model_args.sft_model_path),
                **self.config_kwargs
            )
            model = PeftModel.from_pretrained(model, self.model_args.reward_lora_path)
            # load v_head 
            model_state_dict = torch.load(os.path.join(self.model_args.reward_lora_path, 'adapter_model.bin'))

            model.v_head.load_state_dict({
                    "summary.weight": model_state_dict["v_head.summary.weight"],
                    "summary.bias": model_state_dict["v_head.summary.bias"]
                })
            model = model.merge_and_unload()
        
        model.requires_grad_(False)
        print('*********************rm*******************')
        print_trainable_params(model)
        return model 
    
    def _init_actor(self):
        model = self._create_sft_model()

        if self.model_args.actor_peft_path is not None:
            model = PeftModel.from_pretrained(model, self.model_args.actor_peft_path)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                fan_in_fan_out=True,
                inference_mode=False,
                target_modules=self.training_args.actor_lora_target.split(','),
                r=self.training_args.actor_lora_rank,
                lora_alpha=self.training_args.actor_lora_alpha,
                lora_dropout=self.training_args.actor_lora_dropout,
                modules_to_save=self.training_args.actor_modules_to_save.split(',') if self.training_args.actor_modules_to_save is not None else None
            )

            model = get_peft_model(model, peft_config=lora_config)
        print('*********************actor*******************')
        print_trainable_params(model)
        
        return model 

    def _init_critic(self):
        model = self._create_rm_model()
        v_head_state_dict = model.v_head.state_dict()
        
        if self.model_args.critic_peft_path is not None:
            model = PeftModel.from_pretrained(model, self.model_args.critic_peft_path, adapter_name="default")
            # load v_head 
            model_state_dict = torch.load(os.path.join(self.model_args.critic_peft_path, 'adapter_model.bin'))
            model.v_head.load_state_dict({
                "summary.weight": model_state_dict["v_head.summary.weight"],
                "summary.bias": model_state_dict["v_head.summary.bias"]
            })
        else:
            lora_config = LoraConfig(
                inference_mode=False,
                target_modules=self.training_args.critic_lora_target.split(','),
                r=self.training_args.critic_lora_rank,
                lora_alpha=self.training_args.critic_lora_alpha,
                lora_dropout=self.training_args.critic_lora_dropout,
            )
            model = PeftModel(model, peft_config=lora_config, adapter_name="default")

        model.register_buffer("reward_head_weight", v_head_state_dict["summary.weight"])
        model.register_buffer("reward_head_bias", v_head_state_dict["summary.bias"])
        model.register_buffer("critic_head_weight", v_head_state_dict["summary.weight"])
        model.register_buffer("critic_head_bias", v_head_state_dict["summary.bias"])

        for name, param in model.named_parameters():
            if 'v_head' in name:
                param.requires_grad = True 
                
        print('*********************critic*******************')
        print_trainable_params(model)

        return model 

 
