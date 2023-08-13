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

        accepts_ids, accepts_labels, rejects_ids = [], [], []
        for instance in instances:
            length = len(instance["input_ids"]) // 2 
            accepts_id = instance["input_ids"][:length]
            rejects_id = instance["input_ids"][length:]
            accepts_label = instance["labels"][:length]

            accepts_ids.append(torch.LongTensor(accepts_id))
            accepts_labels.append(torch.LongTensor(accepts_label))
            rejects_ids.append(torch.LongTensor(rejects_id))
            
        accepts_input_ids = torch.nn.utils.rnn.pad_sequence(
            accepts_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        accepts_labels = torch.nn.utils.rnn.pad_sequence(accepts_labels, batch_first=True, padding_value=-100)
        rejects_input_ids = torch.nn.utils.rnn.pad_sequence(
            rejects_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        return dict(
            accepts_input_ids=accepts_input_ids, 
            accepts_labels=accepts_labels, 
            accepts_attention_mask=accepts_input_ids.ne(self.tokenizer.pad_token_id).long(),
            rejects_input_ids=rejects_input_ids,
            rejects_attention_mask=rejects_input_ids.ne(self.tokenizer.pad_token_id).long(),
        )


class PPODataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:

        input_ids = [torch.LongTensor(instance["input_ids"]).flip(0) for instance in instances] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        return dict(
            input_ids=input_ids.flip(1),   
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).flip(1),
        )   # prompt 倒序 

        




