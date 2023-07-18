import torch
import transformers
from transformers import DataCollatorWithPadding, BatchEncoding, PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Optional, Dict, Sequence, Union, List
from dataclasses import dataclass
from typing import Any, List, Union, Optional, Dict

IGNORE_INDEX=-100

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([torch.LongTensor(instance[key]) for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).long(),
        )


class RMDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, instances: Sequence[Dict[str, Union[torch.Tensor, Sequence[int]]]]) -> Dict[str, torch.Tensor]:
        accept_ids, reject_ids = tuple([torch.LongTensor(instance[key]) for instance in instances] for key in ("accept_ids", "reject_ids"))

        accept_ids = torch.nn.utils.rnn.pad_sequence(
            accept_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        reject_ids = torch.nn.utils.rnn.pad_sequence(
            reject_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        return dict(
            accept_ids=accept_ids,
            reject_ids=reject_ids,
            accept_attention_mask=accept_ids.ne(self.tokenizer.pad_token_id).long(),
            reject_attention_mask=reject_ids.ne(self.tokenizer.pad_token_id).long(),
        )


class RLHFDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.tokenizer.pad_token_id
        input_ids = [torch.LongTensor(instance["input_ids"].flip(1)) for instance in instances] ## flip 1 or 0 ?????
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        return dict(
            input_ids=input_ids.flip(1),   
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).flip(1),
        )   # prompt 倒序 

    



