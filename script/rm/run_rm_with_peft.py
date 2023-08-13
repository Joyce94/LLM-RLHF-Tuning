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
from utils.metrics import compute_metrics
from utils.trainer import PeftTrainer,RMPeftTrainer
from trl import AutoModelForCausalLMWithValueHead
from utils.data_collator import RMDataCollatorWithPadding

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaTokenizer, AutoModelForCausalLMWithValueHead),
    "auto": (AutoConfig, AutoTokenizer, AutoModelForCausalLMWithValueHead),
}
IGNORE_INDEX = -100

def main():
    
    model_args, data_args, training_args = parser_arguments(logger)
    transformers.set_seed(training_args.seed)

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
        model = PeftModel.from_pretrained(model, model_args.peft_path)
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
        
    for name, param in model.named_parameters():
        if 'v_head' in name:
            param.requires_grad = True 
    model.print_trainable_parameters()
    # trainable params: 19,992,577 || all params: 6,905,487,361 || trainable%: 0.28951724845536214

    def process_tokenize(examples):
        model_inputs = {"input_ids": [], "labels": []} 
        for instruction, input, output in zip(examples['instruction'], examples['input'], examples['output']):
            if input is not None and input != "":
                instruction = instruction + '\n' + input 
            source = tokenizer.encode_plus(text=instruction, add_special_tokens=False)
            accepts = tokenizer.encode_plus(text=output[0], add_special_tokens=False)
            rejects = tokenizer.encode_plus(text=output[1], add_special_tokens=False)

            accepts_ids = source["input_ids"] + [tokenizer.bos_token_id] + accepts["input_ids"] + [tokenizer.eos_token_id]
            accepts_labels = [IGNORE_INDEX] * len(source["input_ids"]) + [tokenizer.bos_token_id] + accepts["input_ids"] + [tokenizer.eos_token_id]
            rejects_ids = source["input_ids"] + [tokenizer.bos_token_id] + rejects["input_ids"] + [tokenizer.eos_token_id]

            if len(accepts_ids) > training_args.max_length:
                accepts_ids = accepts_ids[:training_args.max_length]
                accepts_labels = accepts_labels[:training_args.max_length]
            if len(rejects_ids) > training_args.max_length:
                rejects_ids = rejects_ids[:training_args.max_length]
            
            accepts_length, rejects_length = len(accepts_ids), len(rejects_ids)
            max_length = max(accepts_length, rejects_length)
            
            accepts_ids = accepts_ids + [tokenizer.pad_token_id] * (max_length - accepts_length)
            accepts_labels = accepts_labels + [IGNORE_INDEX] * (max_length - accepts_length)
            rejects_ids = rejects_ids + [tokenizer.pad_token_id] * (max_length - rejects_length)
            
            inputs_ids = accepts_ids + rejects_ids
            labels = accepts_labels + [0] * len(rejects_ids)
            
            model_inputs["input_ids"].append(inputs_ids)
            model_inputs["labels"].append(labels)

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
                    cache_dir=data_args.data_cache_dir
                )

                tokenized_data = raw_dataset.map(
                    process_tokenize,
                    batched=True,
                    num_proc=training_args.dataloader_num_workers,
                    remove_columns=["instruction","input","output"],
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
            all_datasets['train'] = raw_train_datasets.map(
                process_tokenize,
                batched=True,
                num_proc=training_args.dataloader_num_workers,
                remove_columns=["instruction","input","output"],
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
                remove_columns=["instruction","input","output"],
                load_from_cache_file=True
            )['train']
        else:
            raise ValueError(
                "Dataset file path is not correct. "
                "You can provide --dataset_dir or provide two files --train_file and --validation_file. "
            )

    trainer = RMPeftTrainer(
        model=model,
        args=training_args,
        train_dataset=all_datasets['train'] if training_args.do_train else None,
        eval_dataset=all_datasets['test'] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=RMDataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        output = trainer.train()
        trainer.log_metrics("train", output.metrics)
        trainer.save_metrics("train", output.metrics)
        trainer.save_state()
        trainer.save_model()


if __name__ == "__main__":
    main()




