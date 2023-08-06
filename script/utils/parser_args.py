import sys,os,logging
import transformers
from transformers import HfArgumentParser,TrainingArguments
from dataclasses import dataclass, field 
from typing import List, Optional, Tuple
import datasets 

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    tokenizer_name_or_path: Optional[str] = field(default=None)
    
    use_fast_tokenizer: Optional[bool] = field(default=False)
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    model_type: Optional[str] = field(
        default="opt",
        metadata={
            "help": (
                "If training from scratch, pass a model type from the list"
            ),
            "choices": ["auto", "llama"],
        },
    )

    load_in_4bit: bool = field(default=False)
    peft_path : Optional[str] = field(default=None)

    ## for ppo
    reward_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoints of the reward model."}
    )
    reward_lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoints of the reward lora parameters."}
    )
    sft_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoints of the sft model."}
    )
    actor_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoints of the actor model."}
    )
    critic_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoints of the critic model."}
    )

    actor_peft_path : Optional[str] = field(default=None)
    critic_peft_path : Optional[str] = field(default=None)
    

@dataclass
class DataTrainingArguments:
    dataset_dir: Optional[str] = field(default=None)
    pretrain_dataset_dir: Optional[str] = field(default=None)

    data_cache_dir: Optional[str] = field(default="./")

    block_size: Optional[int] = field(
        default=512,
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

    

@dataclass
class FinetuningArguments(TrainingArguments):
    
    dataloader_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_length: Optional[int] = field(default=512)
    
    report_to: Optional[List[str]] = field(
        default=None, 
        metadata={
            "choices": ["wandb"]
        }
    )
    
    ######## for rm 
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_alpha: Optional[float] = field(default=32.)
    lora_target: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Name(s) of target modules to apply LoRA. \
                  LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"up_proj\", \"down_proj\"]"}
    )
    modules_to_save: Optional[str] = field(default=None)

    output_dir: Optional[str] = field(default=None)

    ######## for ppo 
    
    critic_output_dir: Optional[str] = field(default=None)

    max_prompt_length: Optional[int] = field(default=256)
    max_response_length: Optional[int] = field(default=256)
    min_response_length: Optional[int] = field(default=10)

    ds_zero_stage: Optional[int] = field(default=3)
    offload: Optional[bool] = field(default=False)

    actor_lr: Optional[float] = field(
        default=1e-5,
    )
    actor_lora_rank: Optional[int] = field(default=8)
    actor_lora_dropout: Optional[float] = field(default=0.1)
    actor_lora_alpha: Optional[float] = field(default=32.)
    actor_lora_target: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Name(s) of target modules to apply LoRA. \
                  LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"up_proj\", \"down_proj\"]"}
    )
    actor_modules_to_save: Optional[str] = field(default=None)
    actor_weight_decay: Optional[float] = field(default=0.)

    critic_lr: Optional[float] = field(
        default=1e-5,
        metadata={"help": ""}
    )
    critic_lora_rank: Optional[int] = field(default=8)
    critic_lora_dropout: Optional[float] = field(default=0.1)
    critic_lora_alpha: Optional[float] = field(default=32.)
    critic_lora_target: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Name(s) of target modules to apply LoRA. \
                  LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"up_proj\", \"down_proj\"]"}
    )
    critic_modules_to_save: Optional[str] = field(default=None)
    critic_weight_decay: Optional[float] = field(default=0.)
    

    ppo_epochs: Optional[int] = field(default=1)

    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "Batch size (per device) for the training dataloader and generation purpose."})
    per_device_mini_train_batch_size: Optional[int] = field(default=8, metadata={"help": "Batch size (per device) for the training dataloader and generation purpose."})
    mini_data_shuffle: Optional[bool] = field(default=False)

    gamma: Optional[float] = field(default=1., metadata={"help": "Gamma parameter for advantage calculation"})
    lam: Optional[float] = field(default=0.95, metadata={"help": "Lambda parameter for advantage calculation"})

    kl_penalty_beta: Optional[float] = field(default=0.1)
    reward_score_clip: Optional[float] = field(default=None)
    value_clip: Optional[float] = field(default=0.2)
    ratio_clip: Optional[float] = field(default=0.2)
    
    entropy_beta: Optional[float] = field(default=0.)
    kl_loss_alpha: Optional[float] = field(default=0.)

    actor_loss_weight: Optional[float] = field(default=1.)
    critic_loss_weight: Optional[float] = field(default=1.)
    pretrain_loss_weight: Optional[float] = field(default=1.)
    pretrain_warmup_steps: Optional[int] = field(default=None)



def parser_arguments(logger) -> Tuple[ModelArguments, DataTrainingArguments, FinetuningArguments]:

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FinetuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(f"Model args: {model_args}")
    logger.warning(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    return model_args, data_args, training_args















