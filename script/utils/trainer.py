import os,sys,logging,re 
from datasets import Dataset
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, Trainer,PreTrainedModel
from transformers.modeling_utils import unwrap_model
from typing import Any, Dict, List, Optional, Tuple, Union
from peft import get_peft_model,get_peft_model_state_dict
from transformers.trainer_pt_utils import nested_detach
import inspect,math 
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler,PPODecorators,convert_to_scalar,stats_to_np,stack_dicts,logprobs_from_logits
from trl.models import PreTrainedModelWrapper
from peft.tuners.lora import LoraLayer 
import wandb
from tqdm import tqdm
import time 
import warnings
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


# WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_NAME = "adapter_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class PeftTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if not isinstance(self.model, PreTrainedModel):
            if state_dict is None:
                state_dict = self.model.state_dict()
            
            if isinstance(unwrap_model(self.model), PreTrainedModel):       
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                torch.save(get_peft_model_state_dict(self.model, state_dict), os.path.join(output_dir, WEIGHTS_NAME))

            try:
                unwrap_model(self.model).peft_config.save_pretrained(output_dir)
            except AttributeError:
                unwrap_model(self.model).peft_config['default'].save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


class RMPeftTrainer(PeftTrainer):

    def get_state_dict(self, model):
        pretrained_model_state_dict = model.pretrained_model.state_dict()
        v_head_state_dict = model.v_head.state_dict()
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict 

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if not isinstance(self.model, PreTrainedModel):
            if state_dict is None:
                state_dict = self.get_state_dict(self.model)
            
            if isinstance(unwrap_model(self.model), PreTrainedModel):     
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                adapter_state_dict = get_peft_model_state_dict(self.model, state_dict)

                ### add v_head (v_head not in modules_to_save)
                v_head_state_dict = self.model.v_head.state_dict()
                for k, v in v_head_state_dict.items():
                    adapter_state_dict[f"v_head.{k}"] = v

                torch.save(adapter_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                
            try:
                unwrap_model(self.model).peft_config.save_pretrained(output_dir)
            except AttributeError:
                unwrap_model(self.model).peft_config['default'].save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # self._signature_columns += ["input_ids"]
            if "input_ids" not in self._signature_columns:
                self._signature_columns += ["input_ids"]
                logger.warning(
                    "The following columns have been ignored : input_ids"
                )
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

    def compute_loss(self, model, inputs, return_outputs=False):
        logits, _, value = model(**inputs)
        attention_mask = inputs["attention_mask"]
        value = value * attention_mask
        accept, reject = value[:, :self.args.max_length // 2], value[:, self.args.max_length // 2:]

        loss = -torch.nn.functional.logsigmoid(accept - reject).mean()
        outputs = {"accept_ids": accept, "reject_ids": reject}
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach()
        
        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        logits = torch.stack(logits, dim=0).mean(dim=2).softmax(dim=0)
        return loss, logits, None 



