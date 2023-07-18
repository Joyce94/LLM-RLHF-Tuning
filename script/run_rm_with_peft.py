import os,sys,torch
import numpy as np
from typing import Dict
import transformers
from utils.parser_args import parser_arguments
from transformers import AutoConfig,AutoTokenizer,LlamaForCausalLM,LlamaTokenizer,Trainer,DataCollatorWithPadding,AutoModelForCausalLM
import logging 
from peft import LoraConfig,PeftModel,TaskType,get_peft_model
from pathlib import Path 
from datasets import load_dataset,concatenate_datasets
from itertools import chain
from utils.metrics import compute_metrics
import math 
from utils.trainer import PeftTrainer,RMPeftTrainer
from trl import AutoModelForCausalLMWithValueHead
from utils.data_collator import RMDataCollatorWithPadding
from deepspeed.compression.helper import recursive_getattr, recursive_setattr

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    "llama": (AutoConfig, LlamaTokenizer, LlamaForCausalLM),
    "auto": (AutoConfig, AutoTokenizer, AutoModelForCausalLM),
}


def main():
    
    model_args, data_args, training_args = parser_arguments(logger)
    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    ## load model 
    if model_args.tokenizer_name_or_path is None:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, use_fast=model_args.use_fast_tokenizer)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id # set as the <unk> token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path
    )
    # for name, module in model.named_modules():
    #     print(name)
    # for param in model.parameters():
    #     if param.requires_grad:
    #         print(param)

    if training_args.peft_path is not None:
        logger.info(f"Load pre-trained model: {training_args.peft_path}" )
        model = PeftModel.from_pretrained(model, training_args.peft_path)
    else:
        logger.info("Init new peft model")
        lora_config = LoraConfig(
            inference_mode=False,
            target_modules=training_args.lora_target.split(','),
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            modules_to_save=training_args.modules_to_save.split(',') if training_args.modules_to_save is not None else None
        )
        model = get_peft_model(model, peft_config=lora_config)
    model.print_trainable_parameters()
    # for name, module in model.named_modules():
    #     print(name)
    # model = model.half()

    def process_tokenize(examples):
        model_inputs = {"input_ids": []} 
        for instruction, input, output in zip(examples['instruction'], examples['input'], examples['output']):
            if input is not None and input != "":
                instruction = instruction + '\n' + input 
            source_ids = tokenizer.encode_plus(text=instruction, add_special_tokens=False)
            accepts_ids = tokenizer.encode_plus(text=output[0], add_special_tokens=False)
            rejects_ids = tokenizer.encode_plus(text=output[1], add_special_tokens=False)

            accepts_ids = source_ids["input_ids"] + [tokenizer.bos_token_id] + accepts_ids["input_ids"] + [tokenizer.eos_token_id]
            rejects_ids = source_ids["input_ids"] + [tokenizer.bos_token_id] + rejects_ids["input_ids"] + [tokenizer.eos_token_id]

            if len(accepts_ids) > training_args.max_length:
                accepts_ids = accepts_ids[:training_args.max_length]
            else:
                accepts_ids += [tokenizer.pad_token_id] * (training_args.max_length - len(accepts_ids))

            if len(rejects_ids) > training_args.max_length:
                rejects_ids = rejects_ids[:training_args.max_length]
            else:
                rejects_ids += [tokenizer.pad_token_id] * (training_args.max_length - len(rejects_ids))

            input_ids = accepts_ids + rejects_ids
            model_inputs["input_ids"].append(input_ids)
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
            )['train']
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
        data_collator=RMDataCollatorWithPadding(tokenizer=tokenizer)
    )

    if training_args.do_train:
        # with torch.cuda.amp.autocast():
        output = trainer.train()
        trainer.log_metrics("train", output.metrics)
        trainer.save_metrics("train", output.metrics)
        trainer.save_state()
        trainer.save_model()


    if training_args.do_eval:
        metrics = trainer.evaluate()
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()




