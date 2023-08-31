import datetime
import os,sys,torch,logging
import numpy as np
from typing import Dict
import transformers

sys.path.append('..')

from utils.parser_args import parser_arguments
from transformers import AutoConfig,AutoTokenizer,LlamaForCausalLM,LlamaTokenizer,Trainer,AutoModelForCausalLM,get_scheduler,default_data_collator
from peft import LoraConfig,PeftModel,TaskType,get_peft_model
from pathlib import Path 
from datasets import load_dataset,concatenate_datasets
from itertools import chain
from utils.data_collator import PPODataCollatorWithPadding,DataCollatorForSupervisedDataset
from utils.ppo_models import PPOEngine_CO, PPOEngine
from utils.ppo_trainer_with_peft import PPOPeftTrainer

from utils.utils import PROMPT_TEMPLATE


logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaTokenizer, LlamaForCausalLM),
    "auto": (AutoConfig, AutoTokenizer, AutoModelForCausalLM),
}



def process_data(model_args, data_args, training_args, tokenizer):

    def process_tokenize(examples):
        model_inputs = {"input_ids": [], "label_ids": []}
        columns = list(examples.keys())
        template = PROMPT_TEMPLATE[data_args.template]
        
        for index in range(len(examples[columns[0]])):
            if 'prompt' not in columns:
                assert 'instruction' in columns and 'input' in columns and 'output' in columns

                instruction, input, output = examples['instruction'][index], examples['input'][index], examples['output'][index]
                if input is not None and input != "":
                    instruction = instruction + '\n' + input 
                prompt = instruction
                if len(output) > 1:
                    response = output[0]
                else:
                    response = output 
            else:
                assert 'prompt' in columns
                prompt, response = examples['prompt'][index], examples['chosen'][index]
                
            source = template.format_map({'instruction':prompt})
            source_ids = tokenizer.encode(text=source, add_special_tokens=False)
            target_ids = tokenizer.encode(text=response, add_special_tokens=False)

            if len(source_ids) > training_args.max_prompt_length - 1:
                source_ids = source_ids[:training_args.max_prompt_length - 1]
            if len(target_ids) > training_args.max_response_length - 1:
                target_ids = target_ids[:training_args.max_response_length - 1]
            
            input_ids = source_ids + [tokenizer.bos_token_id]    
            labels = target_ids + [tokenizer.bos_token_id]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["label_ids"].append(labels)
        return model_inputs

    logger.info("process prompt datasets")
    with training_args.main_process_first(desc="process prompt datasets"):
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

            # all_datasets = all_datasets.train_test_split(test_size=data_args.split_ratio)

    
    def process_tokenize_for_pt(examples):
        text_input_ids = tokenizer(examples["text"])["input_ids"]
        concatenated_ids = list(chain(*text_input_ids))
        total_length = len(concatenated_ids)
        if total_length >= data_args.block_size:
            total_length = (total_length // data_args.block_size) * data_args.block_size
        result = [concatenated_ids[i : i + data_args.block_size] for i in range(0, total_length, data_args.block_size)]
        return {"input_ids": result, "label_ids": result.copy()}


    def process_tokenize_for_sft(examples):
        template = PROMPT_TEMPLATE[data_args.template]
        model_inputs = {"input_ids": [], "label_ids": []}
        columns = list(examples.keys())
        
        for index in range(len(examples[columns[0]])):
            if 'prompt' not in columns:
                assert 'instruction' in columns and 'input' in columns and 'output' in columns

                instruction, input, output = examples['instruction'][index], examples['input'][index], examples['output'][index]
                if input is not None and input != "":
                    instruction = instruction + '\n' + input 
                prompt = instruction
                if len(output) > 1:
                    response = output[0]
                else:
                    response = output 
            else:
                assert 'prompt' in columns
                prompt, response = examples['prompt'][index], examples['chosen'][index]
                
            source = template.format_map({'instruction':prompt})
            source_ids = tokenizer.encode(text=source, add_special_tokens=False)
            target_ids = tokenizer.encode(text=response, add_special_tokens=False)

            if len(source_ids) > training_args.max_prompt_length - 1:
                source_ids = source_ids[:training_args.max_prompt_length - 1]
            if len(target_ids) > training_args.max_response_length - 1:
                target_ids = target_ids[:training_args.max_response_length - 1]
                
            input_ids = source_ids + [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]
            labels = [IGNORE_INDEX] * len(source_ids) + [tokenizer.bos_token_id] + target_ids + [tokenizer.eos_token_id]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["label_ids"].append(labels)
        
        return model_inputs


    extra_datasets = []
    if data_args.extra_dataset_dir is not None:
        logger.info("process extra data")
        with training_args.main_process_first(desc="process extra data"):
            path = Path(data_args.extra_dataset_dir)
            if training_args.extra_dataset_type == 'sft':
                files = [file.name for file in path.glob("*.json")]
                for file in files:
                    data_path = os.path.join(path, file)
                    raw_dataset = load_dataset(
                        "json",
                        data_files=data_path,
                    )
                    columns = list(raw_dataset.column_names.values())[0]
                    tokenized_data = raw_dataset.map(
                        process_tokenize_for_sft,
                        batched=True,
                        num_proc=training_args.dataloader_num_workers,
                        remove_columns=columns,
                    )
                    extra_datasets.append(tokenized_data['train'])
                    
            else:
                files = [file.name for file in path.glob("*.txt")]
                for file in files:
                    data_path = os.path.join(path, file)
                    raw_dataset = load_dataset(
                        "text",
                        data_files=data_path
                    )
                    tokenized_data = raw_dataset.map(
                        process_tokenize_for_pt,
                        batched=True,
                        num_proc=training_args.dataloader_num_workers,
                        remove_columns="text"
                    )
                    extra_datasets.append(tokenized_data['train'])
            
            if len(extra_datasets) == 1:
                extra_datasets = extra_datasets[0]
            else:
                extra_datasets = concatenate_datasets(extra_datasets)

    return all_datasets, extra_datasets 


def main():
    
    model_args, data_args, training_args = parser_arguments(logger)
    transformers.set_seed(training_args.seed)
    
    config_class, tokenizer_class, model_class = MODEL_CLASSES[model_args.model_type]
    tokenizer = tokenizer_class.from_pretrained(model_args.sft_model_path, use_fast=model_args.use_fast_tokenizer)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id # set as the <unk> token

    all_datasets, extra_datasets = process_data(model_args, data_args, training_args, tokenizer)
    
    logger.info("training")

    data_collator = PPODataCollatorWithPadding(tokenizer)
    if data_args.extra_dataset_dir is not None:
        if training_args.extra_dataset_type == 'sft':
            extra_data_collator = DataCollatorForSupervisedDataset(tokenizer)
        else:
            extra_data_collator = default_data_collator


    ## load model 
    logger.info("load model")
    if training_args.use_co_model:
        ppo_engine = PPOEngine_CO(model_args, training_args)
    else:
        ppo_engine = PPOEngine(model_args, training_args)

    trainer = PPOPeftTrainer(
        args = training_args, 
        ppo_engine = ppo_engine,
        train_dataset = all_datasets,
        data_collator = data_collator,
        tokenizer = tokenizer,
        extra_train_dataset = extra_datasets if data_args.extra_dataset_dir is not None else None,
        extra_data_collator = extra_data_collator if data_args.extra_dataset_dir is not None else None,
        
    )
    
    if training_args.do_train:
        trainer.train()


if __name__ == "__main__":
    main()

