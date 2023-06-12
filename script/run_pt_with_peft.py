import os,sys
import transformers
from utils import parser_arguments
from transformers import AutoConfig,AutoTokenizer,LlamaForCausalLM,LlamaTokenizer,Trainer,DataCollatorWithPadding
import logging 
from peft import LoraConfig,PeftModel,TaskType,get_peft_model
from pathlib import Path 
from datasets import load_dataset,concatenate_datasets
from itertools import chain
from utils.metrics import compute_metrics
import math 
from utils.trainer import PeftTrainer

logger = logging.getLogger(__name__)


def main():

    model_args, data_args, training_args = parser_arguments(logger)
    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    ## load model 
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
    }
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id # set as the <unk> token

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        torch_dtype=model_args.torch_dtype,
        low_cpu_mem_usage=True
    )

    model.resize_token_embeddings(len(tokenizer))

    logger.info("Init new peft model")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=training_args.lora_target.split(','),
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout
    )

    model = get_peft_model(model, peft_config=lora_config)
    model.print_trainable_parameters()

    def process_tokenize(examples):
        text_input_ids = tokenizer(examples["text"])["input_ids"]
        concatenated_ids = list(chain(*text_input_ids))
        total_length = len(concatenated_ids)
        if total_length >= data_args.block_size:
            total_length = (total_length // data_args.block_size) * data_args.block_size
        result = [concatenated_ids[i : i + data_args.block_size] for i in range(0, total_length, data_args.block_size)]
        return {"input_ids": result, "labels": result.copy()}

    ### process_dataset
    logger.info("process datasets")
    with training_args.main_process_first(desc="process datasets"):
        all_datasets = []
        path = Path(data_args.dataset_dir)
        files = [file.name for file in path.glob("*.txt")]
        for file in files:
            data_path = os.path.join(path, file)
            raw_dataset = load_dataset(
                "text",
                data_files=data_path,
                cache_dir=data_args.data_cache_dir
            )

            tokenized_data = raw_dataset.map(
                process_tokenize,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns="text",
                load_from_cache_file=True
            )
            all_datasets.append(tokenized_data['train'])
        if len(all_datasets) == 1:
            all_datasets = all_datasets[0]
        else:
            all_datasets = concatenate_datasets(all_datasets)

        all_datasets = all_datasets.train_test_split(test_size=data_args.split_ratio)

    ### training
    trainer = PeftTrainer(
        model=model,
        args=training_args,
        train_dataset=all_datasets['train'] if training_args.do_train else None,
        eval_dataset=all_datasets['test'] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics if training_args.do_eval else None
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()


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










