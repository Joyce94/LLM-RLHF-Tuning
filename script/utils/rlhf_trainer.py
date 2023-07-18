import os,sys
import torch
import torch.nn.functional as F
from transformers import Trainer,PreTrainedModel
from transformers.modeling_utils import unwrap_model
from typing import Any, Dict, List, Optional, Tuple, Union
from peft import get_peft_model,get_peft_model_state_dict

WEIGHTS_NAME = "adapter_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class PPOTrainer():
    def __init__(
        self, 
        rlhf_engine,
        model_args, 
        training_args,
        train_dataloader,
        eval_dataloader,
        
        pt_train_dataloader,
        pt_eval_dataloader

    ):
        self.rlhf_engint = rlhf_engine
        self.sft_model = rlhf_engine.sft_model
        self.rm_model = rlhf_engine.rm_model
        self.actor_model = rlhf_engine.actor_model
        self.critic_model = rlhf_engine.critic_model
        self.tokenizer = rlhf_engine.tokenizer

        self.model_args = model_args
        self.training_args = training_args

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.pt_train_dataloader = pt_train_dataloader
        self.pt_eval_dataloader = pt_eval_dataloader 


        self.num_train_epochs = training_args.num_train_epochs
        self.num_train_rl_epochs = training_args.num_train_rl_epochs
        self.max_prompt_length = training_args.max_prompt_length

        self.kl_penalty_beta = training_args.kl_penalty_beta
        self.reward_clip = training_args.reward_clip

        self.pt_weight = training_args.pt_weight


    def train(self):
        for epoch in range(self.num_train_epochs):
            if self.pt_train_dataloader is not None:
                for step, (batch_rlhf_data, batch_pt_data) in enumerate(zip(self.train_dataloader, self.pt_train_dataloader)):
                    output = self.generate(batch_rlhf_data)
                    for rl_epoch in range(self.num_train_rl_epochs):
                        loss = self.train_step(output, batch_pt_data)

            else:
                for step, batch_rlhf_data in enumerate(self.train_dataloader):
                    output = self.generate(batch_rlhf_data)
                    for rl_epoch in range(self.num_train_rl_epochs):
                        loss = self.train_step(output)
        return loss 
    

    # @torch.no_grad()
    # def evaluate(self):
    #     self.actor_model.eval()
    #     self.critic_model.eval()

    #     if self.pt_eval_dataloader is not None:
    #         for step, (batch_rlhf_data, batch_pt_data) in enumerate(zip(self.eval_dataloader, self.pt_eval_dataloader)):
    #             output = self.generate(batch_rlhf_data)



    def train_step(self, data, pt_data=None):
        self.actor_model.train()
        self.critic_model.train()

        old_values = data["values"] * data["responses_mask"]    ## responses_mask or query_ans_mask ????
        start = data["inputs"].shape[1] - 1
        with torch.no_grad():
            rewards_with_kl_penalty = self.compute_rewards(data["sft_log_probs"], data["actor_log_probs"], data["rewards"], data["responses_mask"])
            old_rewards = rewards_with_kl_penalty * data["responses_mask"]
            advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)

        actor_logits = self.actor_model(input_ids=data["responses"], attention_mask=data["responses_mask"]).logits
        actor_log_probs = self.get_log_probs(actor_logits[:, :-1, :], data["responses"][:, 1:])
        values = self.critic_model(input_ids=data["responses"], attention_mask=data["responses_mask"]).logits[:, :-1]

        actor_loss = self.actor_loss_fn(actor_log_probs[:, start:], data["actor_log_probs"][:, start:], advantages, data["responses_mask"][:, start:])
        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        critic_loss = self.critic_loss_fn(values[:, start:], old_values[:, start:], returns, data["responses_mask"][:, start:])
        self.critic_model.backward(critic_loss)
        self.critic_model.step()

        if pt_data is not None:
            pt_loss = self.actor_model(**pt_data, use_cache=False).loss
            self.actor_model.backward(self.pt_weight * pt_loss)
            self.actor_model.step()
            return actor_loss, critic_loss, pt_loss

        return actor_loss, critic_loss


    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        # Adopted from https://github.com/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/ppo_trainer.py
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss


    def critic_loss_fn(self, values, old_values, returns, mask):
        # Adopted from https://github.com/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/ppo_trainer.py
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss


    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/ppo_trainer.py
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]  # TD error 
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns


    def compute_rewards(self, sft_log_probs, actor_log_probs, rewards, responses_mask):
        batch_size = actor_log_probs.shape[0]
        rewards_with_kl_penalty = -self.kl_penalty_beta * (actor_log_probs - sft_log_probs)
        rewards_clip = torch.clamp(rewards, -self.reward_clip, self.reward_clip)

        response_len = responses_mask[:, self.max_prompt_length - 1:].sum(1) + self.max_prompt_length
        for i in range(batch_size):
            rewards_with_kl_penalty[i, self.max_prompt_length - 1 : response_len[i]][-1] += rewards_clip[i]
        return rewards_with_kl_penalty


    def get_rewards(self, inputs, values):
        batch_size = inputs.shape[0]
        len = inputs.shape[1]
        rewards = []
        for i in batch_size:
            input = inputs[i]
            response_len = torch.ne(input[self.max_prompt_length:], self.tokenizer.pad_token_id).sum().item()
            # response_len = (input[self.max_prompt_length:] == self.tokenizer.pad_token_id).sum().item()
            end_index = response_len + self.max_prompt_length 
            rewards.append(values[end_index - 1])
        return rewards
    

    def get_log_probs(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)  
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)) 
        return log_probs_labels.squeeze(-1)


    @torch.no_grad()
    def generate(self, inputs):
        self.actor_model.eval()
        self.critic_model.eval()

        with torch.no_grad():
            sequence = self.actor_model.module.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            sequence_mask = sequence.ne(self.tokenizer.pad_token_id).long()
            actor_logits = self.actor_model(sequence, attention_mask=sequence_mask).logits.detach()
            sft_logits = self.sft_model(sequence, attention_mask=sequence_mask).logits.detach()

            values = self.critic_model(sequence, attention_mask=sequence_mask).logits.detach()[:, :-1] # ??? logits=pooled_logits  # (bz, max_len-1) 
            ## 获取每句话的最后一个位置的分数作为reward (bz,)
            r_values = self.rm_model(sequence, attention_mask=sequence_mask).logits.detach()[:, :-1]
            rewards = self.get_rewards(inputs["input_ids"], r_values)

        return dict(
            inputs=inputs["input_ids"],
            inputs_mask=inputs["attention_mask"],
            sft_log_probs=self.get_log_probs(sft_logits[:, :-1, :], sequence[:, 1:]),
            actor_log_probs=self.get_log_probs(actor_logits[:, :-1, :], sequence[:, 1:]),
            rewards=rewards,
            values=values,
            responses=sequence,
            responses_mask=sequence_mask
        )


    def _save(self, model, output_dir=None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not isinstance(model, PreTrainedModel):
            if state_dict is None:
                state_dict = model.state_dict()
            
            if isinstance(unwrap_model(model), PreTrainedModel):       ######### 
                unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                # torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                torch.save(get_peft_model_state_dict(model, state_dict), os.path.join(output_dir, WEIGHTS_NAME))
            ##### add code 
            try:
                unwrap_model(model).peft_config.save_pretrained(output_dir)
            except AttributeError:
                unwrap_model(model).peft_config['default'].save_pretrained(output_dir)
        else:
            model.save_pretrained(output_dir, state_dict=state_dict)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def save_model(self):
        self._save(self.actor_model, output_dir=self.model_args.output_dir)
        self._save(self.critic_model, output_dir=self.model_args.critic_output_dir)

