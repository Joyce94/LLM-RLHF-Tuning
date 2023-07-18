import os,sys,torch
import numpy as np
from typing import Dict
import transformers
from utils.rlhf_parser_args import parser_arguments
from transformers import AutoConfig,AutoTokenizer,LlamaForCausalLM,LlamaTokenizer,Trainer,DataCollatorWithPadding,AutoModelForCausalLM,LlamaForSequenceClassification,AutoModelForSequenceClassification
import logging 
from peft import LoraConfig,PeftModel,TaskType,get_peft_model
from pathlib import Path 
from datasets import load_dataset,concatenate_datasets
from itertools import chain
from utils.metrics import compute_metrics
import math 
from utils.trainer import PeftTrainer,RMPeftTrainer
from trl import AutoModelForCausalLMWithValueHead
from utils.data_collator import RLHFDataCollatorWithPadding
from utils.rlhf_engine import RLHFEngine
from utils.rlhf_trainer import PPOTrainer
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler


logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def main():
    
    model_args, data_args, training_args = parser_arguments(logger)
    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    ## load model 
    rlhf_engine = RLHFEngine(model_args, training_args)

    def process_tokenize(examples):
        PROMPT_TEMPLATE = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        )
        model_inputs = {"input_ids": [], "labels": []}
        for instruction, input, output in zip(examples['instruction'], examples['input'], examples['output']):
            if input is not None and input != "":
                instruction = instruction + '\n' + input 
            source = PROMPT_TEMPLATE.format_map({'instruction':instruction})
            source_ids = rlhf_engine.tokenizer.encode(text=source, add_special_tokens=False)
            target_ids = rlhf_engine.tokenizer.encode(text=output, add_special_tokens=False)

            input_ids = source_ids + [rlhf_engine.tokenizer.bos_token_id]       ## ???????? bos or eos  left padding ? 
            labels = target_ids + [rlhf_engine.tokenizer.bos_token_id]

            model_inputs["input_ids"].append(torch.LongTensor(input_ids))
            model_inputs["labels"].append(torch.LongTensor(labels))
        
        return model_inputs

    logger.info("process rlhf datasets")
    with training_args.main_process_first(desc="process rlhf datasets"):
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

                tokenized_data = raw_dataset.shuffle().map(
                    process_tokenize,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
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
            all_datasets['train'] = raw_train_datasets.shuffle().map(
                process_tokenize,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=["instruction","input","output"],
                load_from_cache_file=True
            )
            raw_valid_datasets = load_dataset(
                "json",
                data_files=data_args.validation_file,
                cache_dir=data_args.data_cache_dir
            )
            all_datasets['test'] = raw_valid_datasets.shuffle().map(
                process_tokenize,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=["instruction","input","output"],
                load_from_cache_file=True
            )
        else:
            raise ValueError(
                "Dataset file path is not correct. "
                "You can provide --dataset_dir or provide two files --train_file and --validation_file. "
            )
    
    def process_tokenize_for_pt(examples):
        text_input_ids = rlhf_engine.tokenizer(examples["text"])["input_ids"]
        concatenated_ids = list(chain(*text_input_ids))
        total_length = len(concatenated_ids)
        if total_length >= data_args.block_size:
            total_length = (total_length // data_args.block_size) * data_args.block_size
        result = [concatenated_ids[i : i + data_args.block_size] for i in range(0, total_length, data_args.block_size)]
        return {"input_ids": result, "labels": result.copy()}

    if data_args.pretrain_dataset_dir is not None:
        logger.info("process pretrain data")
        with training_args.main_process_first(desc="process pretrain data"):
            pt_datasets = []
            path = Path(data_args.pretrain_dataset_dir)
            files = [file.name for file in path.glob("*.txt")]
            for file in files:
                data_path = os.path.join(path, file)
                raw_dataset = load_dataset(
                    "text",
                    data_files=data_path
                )

                tokenized_data = raw_dataset.shuffle().map(
                    process_tokenize_for_pt,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns="text"
                )
                pt_datasets.append(tokenized_data['train'])
            if len(pt_datasets) == 1:
                pt_datasets = pt_datasets[0]
            else:
                pt_datasets = concatenate_datasets(pt_datasets)
            pt_datasets = pt_datasets.train_test_split(test_size=data_args.split_ratio)


    rlhf_data_collator = RLHFDataCollatorWithPadding(rlhf_engine.tokenizer)
    pt_data_collator = DataCollatorWithPadding(rlhf_engine.tokenizer)

    def create_dataloader(dataset, data_collator):
        if training_args.local_rank == -1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)

        dataloader = DataLoader(
                dataset,
                batch_size=training_args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=data_collator,
                num_workers=data_args.preprocessing_num_workers)
        return dataloader
    
    trainer = PPOTrainer(
        rlhf_engine,
        model_args,
        training_args,
        train_dataloader=create_dataloader(all_datasets['train'], rlhf_data_collator) if training_args.do_train else None,
        eval_dataloader=create_dataloader(all_datasets['test'], rlhf_data_collator) if training_args.do_eval else None,
        pt_train_dataloader=create_dataloader(pt_datasets['train'], pt_data_collator) if training_args.do_train else None,
        pt_eval_dataloader=create_dataloader(pt_datasets['test'], pt_data_collator) if training_args.do_eval else None
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()



if __name__ == "__main__":
    main()

