import os,re,sys
import torch 
from transformers import AutoConfig,AutoTokenizer,LlamaForCausalLM,LlamaTokenizer,Trainer,DataCollatorWithPadding,AutoModelForCausalLM,AutoModelForSequenceClassification,LlamaForSequenceClassification
import logging 
from peft import LoraConfig,PeftModel,TaskType,get_peft_model
from trl import AutoModelForCausalLMWithValueHead,create_reference_model


MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaTokenizer, LlamaForCausalLM),
    "auto": (AutoConfig, AutoTokenizer, AutoModelForCausalLM),
}


'''
1. sft和rm需要增加 冻住参数 的代码吗
2. actor和critic的模型的lr和opt的参数不同，怎么加上 ？？？？？？？？？？？？？

'''
class RLHFEngine():
    def __init__(self, model_args, training_args):
        self.model_args = model_args
        self.training_args = training_args
        self.torch_dtype = (
                model_args.torch_dtype
                if model_args.torch_dtype in ["auto", None]
                else getattr(torch, model_args.torch_dtype)
            )
        self.sft_model, self.tokenizer = self._create_sft_model(self.model_args.sft_model_path) ## ? create_reference_model
        self.rm_model, _ = self._create_rm_model(self.model_args.rm_model_path) # ？ 冻住

        self.actor_model = self._init_actor(self.model_args.sft_model_path)
        self.critic_model = self._init_critic(self.model_args.rm_model_path)

    
    def _create_sft_model(self, model_path):
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=self.model_args.use_fast_tokenizer)
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            from_tf=bool(".ckpt" in model_path),
            config=config,
            cache_dir=self.model_args.sft_cache_dir,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        )
        return model, tokenizer

    def _create_rm_model(self, model_path):
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=self.model_args.use_fast_tokenizer)
        # model = LlamaForCausalLM.from_pretrained(
        #     model_path,
        #     from_tf=bool(".ckpt" in model_path),
        #     config=config,
        #     cache_dir=self.model_args.rm_cache_dir,
        #     torch_dtype=self.torch_dtype,
        #     low_cpu_mem_usage=True
        # )
        # model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model=model)  ## ???


        # #### test 1
        # model = AutoModelForCausalLMWithValueHead.from_pretrained(
        #     model_path,
        #     from_tf=bool(".ckpt" in model_path),
        #     config=config,
        #     cache_dir=self.model_args.rm_cache_dir,
        #     torch_dtype=self.torch_dtype,
        #     low_cpu_mem_usage=True
        # )

        #### test 2 
        # model = AutoModelForCausalLMWithValueHead.from_config(config)
        # model_ckpt_path = os.path.join(model_path, 'pytorch_model.bin')
        # assert os.path.exists(model_ckpt_path), f"Cannot find model checkpoint at {model_ckpt_path}"
        # model.load_state_dict(torch.load(model_ckpt_path))  ## 要加载gpu么

        # #### test 3 
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            from_tf=bool(".ckpt" in model_path),
            config=config,
            cache_dir=self.model_args.rm_cache_dir,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        )

        return model, tokenizer
    
    def _init_actor(self, model_path):
        model, _ = self._create_sft_model(model_path)
        if self.training_args.actor_peft_path is not None:
            model = PeftModel.from_pretrained(model, self.training_args.actor_peft_path)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=self.training_args.actor_lora_target.split(','),
                r=self.training_args.actor_lora_rank,
                lora_alpha=self.training_args.actor_lora_alpha,
                lora_dropout=self.training_args.actor_lora_dropout,
                modules_to_save=self.training_args.actor_modules_to_save.split(',') if self.training_args.actor_modules_to_save is not None else None
            )

            model = get_peft_model(model, peft_config=lora_config)
        model.print_trainable_parameters()

        ## scheduler,optimizer 
        return model 

    def _init_critic(self, model_path):
        model, _ = self._create_rm_model(model_path)
        if self.training_args.critic_peft_path is not None:
            model = PeftModel.from_pretrained(model, self.training_args.critic_peft_path)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                target_modules=self.training_args.critic_lora_target.split(','),
                r=self.training_args.critic_lora_rank,
                lora_alpha=self.training_args.critic_lora_alpha,
                lora_dropout=self.training_args.critic_lora_dropout,
                modules_to_save=self.training_args.critic_modules_to_save.split(',') if self.training_args.critic_modules_to_save is not None else None
            )
            model = get_peft_model(model, peft_config=lora_config)

        model.print_trainable_parameters()
        return model 























