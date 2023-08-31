import os,sys,torch,logging,math
import numpy as np
from typing import Dict
import transformers
from transformers import AutoConfig,AutoTokenizer,LlamaForCausalLM,LlamaTokenizer,Trainer,DataCollatorWithPadding,AutoModelForCausalLM,BitsAndBytesConfig

sys.path.append("..")
from peft import LoraConfig,PeftModel,TaskType,get_peft_model
from pathlib import Path 
from datasets import load_dataset,concatenate_datasets
from itertools import chain
from utils.parser_args import parser_arguments
from utils.metrics import compute_metrics_for_pair
from utils.trainer import PeftTrainer,RMPeftTrainer
from trl import AutoModelForCausalLMWithValueHead
from utils.data_collator import PairDataCollatorWithPadding
from utils.utils import PROMPT_TEMPLATE


logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaTokenizer, LlamaForCausalLM),
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



def create_model(model_args, data_args, training_args):

    ## load model 
    config_class, tokenizer_class, model_class = MODEL_CLASSES[model_args.model_type]
    if model_args.tokenizer_name_or_path is None:
        tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_args.tokenizer_name_or_path, use_fast=model_args.use_fast_tokenizer)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id # set as the <unk> token

    config_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype),
        "low_cpu_mem_usage": True
    }
    if model_args.load_in_4bit:
        config_kwargs["load_in_4bit"] = True
        config_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    model = model_class.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        **config_kwargs
    )

    if model_args.peft_path is not None:
        logger.info(f"Load pre-trained model: {model_args.peft_path}" )
        model = PeftModel.from_pretrained(model, model_args.peft_path, is_trainable=True)
        
    else:
        logger.info("Init new peft model")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,      
            inference_mode=False,
            target_modules=training_args.lora_target.split(','),
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config=lora_config)
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    if model_args.peft_path is not None:
        lora_state_dict = torch.load(os.path.join(model_args.peft_path, 'adapter_model.bin'))
        model.v_head.load_state_dict({
                    "summary.weight": lora_state_dict["v_head.summary.weight"],
                    "summary.bias": lora_state_dict["v_head.summary.bias"]
                })
        
    print('*********************model*******************')
    print_trainable_params(model)
    
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model, tokenizer
    


def process_data(model_args, data_args, training_args, tokenizer):
    
    def process_tokenize(examples):
        model_inputs = {"input_ids": [], "label_ids": []} 
        columns = list(examples.keys())
        # logger.info(f"columns: {columns}")
        template = PROMPT_TEMPLATE[data_args.template]
        
        for index in range(len(examples[columns[0]])):
            if 'chosen' not in columns or 'rejected' not in columns:
                assert 'instruction' in columns and 'input' in columns and 'output' in columns

                instruction, input, output = examples['instruction'][index], examples['input'][index], examples['output'][index]
                if input is not None and input != "":
                    instruction = instruction + '\n' + input 
                assert len(output) > 1 
                prompt, chosen, rejected = instruction, output[0], output[1]
            else:
                assert 'prompt' in columns and 'rejected' in columns and 'chosen' in columns
                prompt, chosen, rejected = examples['prompt'][index], examples['chosen'][index], examples['rejected'][index]
                
            source = template.format_map({'instruction':prompt})
            source_ids = tokenizer.encode(text=source, add_special_tokens=False)
            accepts_ids = tokenizer.encode(text=chosen, add_special_tokens=False)
            rejects_ids = tokenizer.encode(text=rejected, add_special_tokens=False)

            if len(source_ids) > training_args.max_prompt_length - 1:
                source_ids = source_ids[:training_args.max_prompt_length - 1]
            if len(accepts_ids) > training_args.max_response_length - 1:
                accepts_ids = accepts_ids[:training_args.max_response_length - 1]
            if len(rejects_ids) > training_args.max_response_length - 1:
                rejects_ids = rejects_ids[:training_args.max_response_length - 1]
                
            
            source_accepts_ids = source_ids + [tokenizer.bos_token_id] + accepts_ids + [tokenizer.eos_token_id]
            source_accepts_labels = [IGNORE_INDEX] * len(source_ids) + [tokenizer.bos_token_id] + accepts_ids + [tokenizer.eos_token_id]
            source_rejects_ids = source_ids + [tokenizer.bos_token_id] + rejects_ids + [tokenizer.eos_token_id]
            source_rejects_labels = [IGNORE_INDEX] * len(source_ids) + [tokenizer.bos_token_id] + rejects_ids + [tokenizer.eos_token_id]


            source_accepts_length, source_rejects_length = len(source_accepts_ids), len(source_rejects_ids)
            max_length = max(source_accepts_length, source_rejects_length)
            
            source_accepts_ids = source_accepts_ids + [tokenizer.pad_token_id] * (max_length - source_accepts_length)
            source_accepts_labels = source_accepts_labels + [IGNORE_INDEX] * (max_length - source_accepts_length)
            source_rejects_ids = source_rejects_ids + [tokenizer.pad_token_id] * (max_length - source_rejects_length)
            source_rejects_labels = source_rejects_labels + [IGNORE_INDEX] * (max_length - source_rejects_length)
            
            inputs_ids = source_accepts_ids + source_rejects_ids
            labels = source_accepts_labels + source_rejects_labels 
            
            model_inputs["input_ids"].append(inputs_ids)
            model_inputs["label_ids"].append(labels)

        return model_inputs



    ### process_dataset
    logger.info("process datasets")
    with training_args.main_process_first(desc="process datasets"):
        if data_args.dataset_dir is not None:
            all_datasets = []
            path = Path(data_args.dataset_dir)
            files = [file.name for file in path.glob("*.json")]
            for file in files:
                data_path = os.path.join(path, file)
                raw_dataset = load_dataset(
                    "json",
                    data_files=data_path,
                )
                columns = list(raw_dataset.column_names.values())[0]
                tokenized_data = raw_dataset.map(
                    process_tokenize,
                    batched=True,
                    num_proc=training_args.dataloader_num_workers,
                    remove_columns=columns,
                    load_from_cache_file=True
                )
                all_datasets.append(tokenized_data['train'])
            if len(all_datasets) == 1:
                all_datasets = all_datasets[0]
            else:
                all_datasets = concatenate_datasets(all_datasets)

            all_datasets = all_datasets.train_test_split(test_size=data_args.split_ratio)
        elif data_args.train_file is not None and data_args.validation_file is not None:
            all_datasets = {}
            raw_train_datasets = load_dataset(
                "json",
                data_files=data_args.train_file,
                cache_dir=data_args.data_cache_dir
            )
            columns = list(raw_train_datasets.column_names.values())[0]
            all_datasets['train'] = raw_train_datasets.map(
                process_tokenize,
                batched=True,
                num_proc=training_args.dataloader_num_workers,
                remove_columns=columns,
                load_from_cache_file=True
            )['train']
            raw_valid_datasets = load_dataset(
                "json",
                data_files=data_args.validation_file,
                cache_dir=data_args.data_cache_dir
            )
            all_datasets['test'] = raw_valid_datasets.map(
                process_tokenize,
                batched=True,
                num_proc=training_args.dataloader_num_workers,
                remove_columns=columns,
                load_from_cache_file=True
            )['train']
        else:
            raise ValueError(
                "Dataset file path is not correct. "
                "You can provide --dataset_dir or provide two files --train_file and --validation_file. "
            )

    return all_datasets


def main():
    
    model_args, data_args, training_args = parser_arguments(logger)
    transformers.set_seed(training_args.seed)

    model, tokenizer = create_model(model_args, data_args, training_args)
    all_datasets = process_data(model_args, data_args, training_args, tokenizer)


    trainer = RMPeftTrainer(
        model=model,
        args=training_args,
        train_dataset=all_datasets['train'] if training_args.do_train else None,
        eval_dataset=all_datasets['test'] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=PairDataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics_for_pair,
    )

    if training_args.do_train:
        output = trainer.train()
        trainer.log_metrics("train", output.metrics)
        trainer.save_metrics("train", output.metrics)
        trainer.save_state()
        trainer.save_model()


if __name__ == "__main__":
    main()




