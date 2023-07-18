import os,sys,logging
import torch
from torch import nn
from transformers import Trainer,PreTrainedModel
from transformers.modeling_utils import unwrap_model
from typing import Any, Dict, List, Optional, Tuple, Union
from peft import get_peft_model,get_peft_model_state_dict
from transformers.trainer_pt_utils import nested_detach


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
            
            if isinstance(unwrap_model(self.model), PreTrainedModel):       ######### 
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                # torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                torch.save(get_peft_model_state_dict(self.model, state_dict), os.path.join(output_dir, WEIGHTS_NAME))
            ##### add code 
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

    def compute_loss(self, model, inputs, return_outputs=False):
        accept = model(input_ids=inputs['accept_ids'], attention_mask=inputs['accept_attention_mask'])
        reject = model(input_ids=inputs['reject_ids'], attention_mask=inputs['reject_attention_mask'])
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
        logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T
        labels = torch.zeros(logits.shape[0])

        return loss, logits, labels




