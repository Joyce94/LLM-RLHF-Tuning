import sys,os,logging
import transformers
from transformers import HfArgumentParser,TrainingArguments
from dataclasses import dataclass, field 
from typing import Optional, Tuple
import datasets 



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": ""})
    tokenizer_name_or_path: Optional[str] = field(default=None, metadata={"help": ""})
    cache_dir: Optional[str] = field(default=None, metadata={"help": ""})
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


    def __post_init__(self):
        return 

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

    def __post_init__(self):
        return 

@dataclass
class FinetuningArguments(TrainingArguments):

    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_alpha: Optional[float] = field(default=32.)
    lora_target: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Name(s) of target modules to apply LoRA. Use comma to separate multiple modules. \
                  LLaMA choices: [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"up_proj\", \"down_proj\"], \
                  BLOOM choices: [\"query_key_value\", \"dense\", \"dense_\"]"}
    )


def parser_arguments(logger) -> Tuple[ModelArguments, DataTrainingArguments, FinetuningArguments]:

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FinetuningArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)


    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()


    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    return model_args, data_args, training_args















