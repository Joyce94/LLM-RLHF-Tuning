import os,sys,logging,re 
from datasets import Dataset
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, Trainer,PreTrainedModel,Seq2SeqTrainer
from transformers.modeling_utils import unwrap_model
from typing import Any, Dict, List, Optional, Tuple, Union
from peft import get_peft_model,get_peft_model_state_dict
from transformers.trainer_pt_utils import nested_detach
import inspect,math 
from peft.tuners.lora import LoraLayer 
from peft import PeftModel
from tqdm import tqdm
import time 
import warnings
import shutil
from pathlib import Path
import torch.nn.functional as F


logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = dict()
        
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
                adapter_state_dict = get_peft_model_state_dict(unwrap_model(self.model).pretrained_model, state_dict)

                ### add v_head (v_head not in modules_to_save)
                v_head_state_dict = self.model.v_head.state_dict()
                for k, v in v_head_state_dict.items():
                    adapter_state_dict[f"v_head.{k}"] = v

                torch.save(adapter_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                
            try:
                unwrap_model(self.model).pretrained_model.peft_config.save_pretrained(output_dir)
            except AttributeError:
                unwrap_model(self.model).pretrained_model.peft_config['default'].save_pretrained(output_dir)
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
            logger.info(f"The following columns {self._signature_columns} are accepted.")

            if "input_ids" not in self._signature_columns:
                self._signature_columns += ["input_ids"]
                logger.warning(
                    "The following columns have been ignored : input_ids"
                )
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            logger.info(f"The following columns {self._signature_columns} are accepted.")
            

    def compute_loss(self, model, inputs, return_outputs=False):

        _, accepts_clm_loss, accepts_value = model(input_ids=inputs["accepts_input_ids"], attention_mask=inputs["accepts_attention_mask"], labels=inputs["accepts_labels"], return_dict=True)
        _, _, rejects_value = model(input_ids=inputs["rejects_input_ids"], attention_mask=inputs["rejects_attention_mask"], return_dict=True)
        
        accepts_labels, rejects_labels = inputs["accepts_labels"], inputs["rejects_labels"]
        accepts_action_masks = accepts_labels.ne(IGNORE_INDEX).long()
        rejects_action_masks = rejects_labels.ne(IGNORE_INDEX).long()
        
        accepts_value = accepts_value * accepts_action_masks
        rejects_value = rejects_value * rejects_action_masks
        
        batch_size = accepts_value.shape[0]
        accepts_seq_lengths = (torch.ne(inputs["accepts_input_ids"], self.tokenizer.pad_token_id).sum(-1) - 1).to(accepts_value.device)
        rejects_seq_lengths = (torch.ne(inputs["rejects_input_ids"], self.tokenizer.pad_token_id).sum(-1) - 1).to(rejects_value.device)
        
        accepts_end_token_value = accepts_value[torch.arange(batch_size, device=accepts_value.device), accepts_seq_lengths]
        rejects_end_token_value = rejects_value[torch.arange(batch_size, device=rejects_value.device), rejects_seq_lengths]
        
        
        if self.args.use_last_reward:
            loss1 = -torch.nn.functional.logsigmoid(accepts_end_token_value - rejects_end_token_value).mean()
        else:
            
            loss1 = -torch.nn.functional.logsigmoid(accepts_value - rejects_value).mean()
            
        loss2 = self.args.clm_loss_weight * accepts_clm_loss
        loss = loss1 + loss2 
        
        outputs = dict(
            accepts_end_token_value=accepts_end_token_value,    # shape: (batch_size,)
            rejects_end_token_value=rejects_end_token_value,    # shape: (batch_size,)
        )

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
        
        for k, v in outputs.items():
            self.metrics[k] = v.mean()
        
        logits = tuple(v for k, v in outputs.items() if k in ["accepts_end_token_value", "rejects_end_token_value"])
        if prediction_loss_only:
            return (loss, None, None)

        logits = torch.stack(logits, dim=1)
        labels = torch.zeros(logits.shape[0]).to(logits.device)
        return loss, logits, labels 


    def log(self, logs):
        if len(self.metrics) > 0:
            for k, v in self.metrics.items():
                logs[f"eval_{k}"] = v.item()
            self.metrics.clear()

        return super().log(logs)
    


class DPOPeftTrainer(PeftTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = dict()
        

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


    def get_log_probs(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)  
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)) 
        return log_probs_labels.squeeze(-1)


    def masked_mean(self, data, mask, dim=None, eps=1e-8):
        data = data * mask
        if dim is not None:
            return data.sum(dim=dim) / (mask.sum(dim=dim) + eps)
        else:
            return data.sum() / (mask.sum() + eps) 
    
    def get_entropy(self, logits, mask):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)  
        entropy = self.masked_mean(-torch.sum(probs * log_probs, dim=-1), mask)
        return entropy 
        
        
    def get_model_output(self, model, inputs, is_ref_model=False):
        
        if is_ref_model:
            if isinstance(model, nn.parallel.DistributedDataParallel):
                with model.module.disable_adapter():
                    accepts_logits = model(input_ids=inputs["accepts_input_ids"], attention_mask=inputs["accepts_attention_mask"], return_dict=True).logits
                    rejects_logits = model(input_ids=inputs["rejects_input_ids"], attention_mask=inputs["rejects_attention_mask"], return_dict=True).logits
            elif isinstance(model, PeftModel):
                with model.disable_adapter():
                    accepts_logits = model(input_ids=inputs["accepts_input_ids"], attention_mask=inputs["accepts_attention_mask"], return_dict=True).logits
                    rejects_logits = model(input_ids=inputs["rejects_input_ids"], attention_mask=inputs["rejects_attention_mask"], return_dict=True).logits 
            else:
                raise AttributeError(
                    f" model object [{model.__class__.__name__}] has no attribute [disable_adapter] "
                )
        else:
            accepts_logits = model(input_ids=inputs["accepts_input_ids"], attention_mask=inputs["accepts_attention_mask"], return_dict=True).logits
            rejects_logits = model(input_ids=inputs["rejects_input_ids"], attention_mask=inputs["rejects_attention_mask"], return_dict=True).logits
            
        accepts_labels, rejects_labels = inputs["accepts_labels"], inputs["rejects_labels"]
        accepts_action_masks = accepts_labels.ne(IGNORE_INDEX).long()
        rejects_action_masks = rejects_labels.ne(IGNORE_INDEX).long()
        
        accepts_log_probs = self.get_log_probs(accepts_logits[:, :-1, :], inputs["accepts_input_ids"][:, 1:])
        rejects_log_probs = self.get_log_probs(rejects_logits[:, :-1, :], inputs["rejects_input_ids"][:, 1:])

        if self.args.average_log_prob:
            accepts_logps = self.masked_mean(accepts_log_probs, accepts_action_masks[:, 1:], dim=-1)
            rejects_logps = self.masked_mean(rejects_log_probs, rejects_action_masks[:, 1:], dim=-1)
        else:
            accepts_logps = (accepts_log_probs * accepts_action_masks[:, 1:]).sum(dim=-1)
            rejects_logps = (rejects_log_probs * rejects_action_masks[:, 1:]).sum(dim=-1)
        
        
        accepts_entropy = self.get_entropy(accepts_logits[:, :-1, :], accepts_action_masks[:, 1:])
        rejects_entropy = self.get_entropy(rejects_logits[:, :-1, :], rejects_action_masks[:, 1:])
        return accepts_entropy, rejects_entropy, accepts_logps, rejects_logps 
    
            
    def compute_loss(self, model, inputs, return_outputs=False):

        accepts_entropy, rejects_entropy, accepts_logps, rejects_logps = self.get_model_output(model, inputs)
        with torch.no_grad():
            ref_accepts_entropy, ref_rejects_entropy, ref_accepts_logps, ref_rejects_logps = self.get_model_output(model, inputs, is_ref_model=True)

        accepts_ratio = self.args.dpo_beta * (accepts_logps - ref_accepts_logps)
        rejects_ratio = self.args.dpo_beta * (rejects_logps - ref_rejects_logps)
        
        pi_ratio = self.args.dpo_beta * (accepts_logps - rejects_logps)
        ref_ratio = self.args.dpo_beta * (ref_accepts_logps - ref_rejects_logps)

        if self.args.reference_free:
            ref_ratio = 0 

        loss = -torch.nn.functional.logsigmoid(pi_ratio - ref_ratio).mean()

        # perplexity 
        outputs = dict(
            accepts_reward=accepts_ratio.detach(),         # shape: (batch_size,)
            rejects_reward=rejects_ratio.detach(),
            pi_ratio=pi_ratio.detach(),
            ref_ratio=ref_ratio.detach(),
            accepts_entropy=accepts_entropy.detach(),
            rejects_entropy=rejects_entropy.detach(),
            ref_accepts_entropy=ref_accepts_entropy.detach(),
            ref_rejects_entropy=ref_rejects_entropy.detach(),
            accepts_ce_loss=-accepts_logps.detach(),
            rejects_ce_loss=-rejects_logps.detach(),
            
        )

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

        for k, v in outputs.items():
            self.metrics[k] = v.mean()
        
        logits = tuple(v for k, v in outputs.items() if k in ["accepts_reward", "rejects_reward"])
        if prediction_loss_only:
            return (loss, None, None)

        logits = torch.stack(logits, dim=1)
        labels = torch.zeros(logits.shape[0]).to(logits.device)
        return loss, logits, labels 


    def log(self, logs):
        if len(self.metrics) > 0:
            for k, v in self.metrics.items():
                logs[f"eval_{k}"] = v.item()
            self.metrics.clear()

        return super().log(logs)
    
    
    