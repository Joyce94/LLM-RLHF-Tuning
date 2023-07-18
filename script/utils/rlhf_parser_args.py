import sys,os,logging
import transformers
from transformers import HfArgumentParser,TrainingArguments
from dataclasses import dataclass, field 
from typing import Optional, Tuple
import datasets 



@dataclass
class ModelArguments:
    use_fast_tokenizer: Optional[bool] = field(default=False, metadata={"help": ""})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    model_type: Optional[str] = field(
        default="auto",
        metadata={
            "help": (
                "If training from scratch, pass a model type from the list"
            ),
            "choices": ["auto", "llama"],
        },
    )
    rm_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoints of the reward model."}
    )
    sft_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoints of the reward model."}
    )

    actor_output_dir: Optional[str] = field(default=None)
    critic_output_dir: Optional[str] = field(default=None)


@dataclass
class DataTrainingArguments:
    dataset_dir: Optional[str] = field(default=None, metadata={"help": ""})
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed stored"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    split_ratio: Optional[float] = field(
        default=0.05,
        metadata={"help": "Proportion of the dataset to include in the development set, should be between 0.0 and 1.0."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    pretrain_dataset_dir: Optional[str] = field(default=None)


@dataclass
class FinetuningArguments(TrainingArguments):
    actor_peft_path : Optional[str] = field(default=None)
    actor_lr: Optional[float] = field(
        default=1e-5,
        metadata={"help": ""}
    )
    actor_lora_rank: Optional[int] = field(default=8)
    actor_lora_dropout: Optional[float] = field(default=0.1)
    actor_lora_alpha: Optional[float] = field(default=32.)
    actor_lora_target: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Name(s) of target modules to apply LoRA. Use comma to separate multiple modules. \
                  LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"up_proj\", \"down_proj\"], \
                  BLOOM choices: [\"query_key_value\", \"dense\", \"dense_\"]"}
    )
    actor_modules_to_save : Optional[str] = field(default=None)
    
    critic_lr: Optional[float] = field(
        default=1e-5,
        metadata={"help": ""}
    )
    critic_peft_path : Optional[str] = field(default=None)
    critic_lora_rank: Optional[int] = field(default=8)
    critic_lora_dropout: Optional[float] = field(default=0.1)
    critic_lora_alpha: Optional[float] = field(default=32.)
    critic_lora_target: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Name(s) of target modules to apply LoRA. Use comma to separate multiple modules. \
                  LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"up_proj\", \"down_proj\"], \
                  BLOOM choices: [\"query_key_value\", \"dense\", \"dense_\"]"}
    )
    critic_modules_to_save : Optional[str] = field(default=None)
    
    max_prompt_length: Optional[int] = field(
        default=256, 
        metadata={"help": ""}
    )
    max_response_length: Optional[int] = field(
        default=256,
        metadata={"help": ""}
    )
    num_train_rl_epochs: Optional[int] = field(default=1)

    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "Batch size (per device) for the training dataloader and generation purpose."})
    per_device_mini_train_batch_size: Optional[int] = field(default=8, metadata={"help": "Batch size (per device) for the training dataloader and generation purpose."})
    
    gamma: Optional[float] = field(default=1, metadata={"help": "Gamma parameter for advantage calculation"})
    lam: Optional[float] = field(default=0.95, metadata={"help": "Lambda parameter for advantage calculation"})

    kl_penalty_beta: Optional[float] = field(default=0.1)
    reward_clip: Optional[float] = field(default=5)

    pt_weight: Optional[float] = field(default=1.)

def parser_arguments(logger) -> Tuple[ModelArguments, DataTrainingArguments, FinetuningArguments]:

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FinetuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
        
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(f"Model args: {model_args}")
    logger.warning(f"Data args: {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    return model_args, data_args, training_args















