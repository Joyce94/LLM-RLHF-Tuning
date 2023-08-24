import re
from typing import List, Tuple
import torch, os, sys, math, logging, random, time, warnings, shutil, copy
from tqdm import tqdm

sys.path.append("..")
from transformers import Trainer,get_scheduler,PreTrainedModel
from torch.utils.data import DataLoader, RandomSampler
from accelerate import Accelerator
from torch.optim import AdamW,Adam
import torch.nn.functional as F
from pathlib import Path
from peft import get_peft_model,get_peft_model_state_dict
import torch.nn as nn 
from datasets import Dataset


logger = logging.getLogger(__name__)

WEIGHTS_NAME = "adapter_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"

class PPOModel(nn.Module):
    # see from: https://github.com/huggingface/accelerate/issues/668
    def __init__(self, actor_model, critic_model):
        super().__init__()
        self.actor_model = actor_model 
        self.critic_model = critic_model 
    
    def forward(self, sequences, extra_inputs=None):
        actor_logits = self.actor_model(**sequences, return_dict=True).logits
        critic_values = self.critic_model(**sequences)[-1][:,:-1]
        if extra_inputs is not None:
            extra_loss = self.actor_model(**extra_inputs, return_dict=True).loss
        else:
            extra_loss = 0.0  
        return actor_logits, critic_values, extra_loss
    
    
class PPOPeftTrainer(Trainer):
    def __init__(
        self, 
        args = None, 
        sft_model = None, 
        reward_model = None,
        actor_model = None,
        critic_model = None,
        data_collator = None,
        train_dataset = None,
        tokenizer = None,
        extra_train_dataset = None,
        extra_data_collator = None, 
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        **kwargs
    ):
        self.args = args 
        self.actor_model = actor_model 
        self.critic_model = critic_model
        self.sft_model = sft_model
        self.reward_model = reward_model

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision='fp16' if self.args.fp16 else None,
            log_with=self.args.report_to,
            
        )
        self.accelerator.init_trackers(
            project_name="ppo_train",
            config=self.args 
        )
        
        self.dataloader = DataLoader(
                                    train_dataset,
                                    batch_size=self.args.per_device_train_batch_size,
                                    collate_fn=data_collator,
                                    num_workers=self.args.dataloader_num_workers,
                                    shuffle=True,
                                    )
        self.dataloader = self.accelerator.prepare(self.dataloader)
        
        if extra_train_dataset is not None:
            self.extra_train_dataloader = DataLoader(
                extra_train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=extra_data_collator,
                num_workers=self.args.dataloader_num_workers,
                shuffle=True,
            )
            self.extra_train_dataloader = self.accelerator.prepare(self.extra_train_dataloader)
        else:
            self.extra_train_dataloader = None 
        
        self.tokenizer = tokenizer
        
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"
        self.device = self.accelerator.device

        self.ppl_loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        self.model = PPOModel(actor_model, critic_model)

        ## create optimizer and scheduler 
        self.optimizer, self.lr_scheduler = optimizers
        if self.optimizer is None:
            self.optimizer = self.create_optimizer(self.actor_model, self.critic_model, self.args.actor_lr, self.args.critic_lr, self.args.actor_weight_decay, self.args.critic_weight_decay)
        
        ## get max_update_steps for lr_scheduler 
        if self.extra_train_dataloader is None:
            self.max_dataloader_iters = len(self.dataloader)
        else:
            self.max_dataloader_iters = min(len(self.dataloader), len(self.extra_train_dataloader))
        self.num_update_steps_per_epoch, self.max_update_steps = self.get_max_update_steps(args, self.max_dataloader_iters)
        
        if self.lr_scheduler is None:
            self.lr_scheduler = self.create_scheduler(self.optimizer, max_update_steps=self.max_update_steps)

        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
        self.ator_model = self.accelerator.unwrap_model(self.model).actor_model 
        self.critic_model = self.accelerator.unwrap_model(self.model).critic_model
        
        
    def get_max_update_steps(self, args, dataloader_nums):
        num_update_steps_per_epoch = dataloader_nums * (args.per_device_train_batch_size / args.per_device_mini_train_batch_size) * args.ppo_epochs / args.gradient_accumulation_steps  
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        
        if args.max_steps > 0:
            max_update_steps = args.max_steps
        else:
            max_update_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        return num_update_steps_per_epoch, max_update_steps
        
        
    def get_parms(self, model, lr, weight_decay, eps=1e-8):
        params = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": weight_decay,
                "lr": lr,
                "eps": eps,
            }
        ]
        return params
    
    
    def create_optimizer(self, actor_model, critic_model, actor_lr, critic_lr, actor_weight_decay, critic_weight_decay):
        params = self.get_parms(actor_model, actor_lr, actor_weight_decay)
        params.extend(self.get_parms(critic_model, critic_lr, critic_weight_decay))

        optimizer = AdamW(params, betas=(0.9,0.95))
        
        return optimizer
    
    
    def create_scheduler(self, optimizer, max_update_steps):
        lr_scheduler = get_scheduler(self.args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=max_update_steps)
        return lr_scheduler


    def masked_mean(self, data, mask, dim=None, eps=1e-8):
        data = data * mask
        if dim is not None:
            return data.sum(dim=dim) / (mask.sum(dim=dim) + eps)
        else:
            return data.sum() / (mask.sum() + eps) 
    
    def masked_var(self, data, mask, dim=None):
        mean = self.masked_mean(data, mask, dim=dim)
        centered_values = data - mean
        var = self.masked_mean(centered_values**2, mask, dim=dim)
        return var


    @torch.no_grad()
    def generate(
        self,
        prompts_ids,
        return_prompt: bool = True,
    ):

        gen_kwargs = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "max_new_tokens": self.args.max_response_length,
            "min_new_tokens": self.args.min_response_length, 
            "_from_model_config": False
        }
        
        if self.actor_model.generation_config._from_model_config:   ### 
            self.actor_model.generation_config._from_model_config = False

        sequences = self.actor_model.generate(inputs=prompts_ids, **gen_kwargs)
        
        if not return_prompt:
            return sequences[:, prompts_ids.shape[1] :]
        
        return sequences


    def process_sequences(self, prompts_ids, responses_ids):
        # seq: [0 0 0 0, prompt, response, 0 0 0 0] change to [prompt, response, 0 0 0 0]
        
        prompts_without_padding, responses_without_padding = [], []
        batch_size = prompts_ids.shape[0]
        for i in range(batch_size):
            response = responses_ids[i]
            prompt = prompts_ids[i] 
            prompt_left_padding_length = (prompt == self.tokenizer.pad_token_id).sum().item()
            response_length = (response != self.tokenizer.pad_token_id).sum().item()
            prompt_without_padding = prompt[prompt_left_padding_length:]
            response_without_padding = response[:response_length]
            
            prompts_without_padding.append(prompt_without_padding.to(self.device))
            responses_without_padding.append(response_without_padding.to(self.device))
        
        
        new_sequences = [torch.cat([q, r]) for q, r in zip(prompts_without_padding, responses_without_padding)]
        sequences = torch.nn.utils.rnn.pad_sequence(
            new_sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        sequences = dict(
            input_ids=sequences.to(self.device),
            attention_mask=sequences.ne(self.tokenizer.pad_token_id).long().to(self.device)
        )
        
        return prompts_without_padding, responses_without_padding, sequences
      
    
    def get_last_reward_score(self, values, responses_mask):
        batch_size = values.shape[0]
        reward_score = []
        for i in range(batch_size):
            value = values[i]
            end_index = responses_mask[i].nonzero()[-1].detach().item()
            reward_score.append(value[end_index])
        return torch.stack(reward_score)
    
    
    def get_log_probs(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)  
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)) 
        return log_probs_labels.squeeze(-1)

    def get_entropy(self, logits, mask):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)  
        entropy = self.masked_mean(-torch.sum(probs * log_probs, dim=-1), mask)
        return entropy 
        
    
    def compute_rewards_with_kl_penalty(self, ref_values, actor_log_probs, ref_log_probs, responses_mask):
        masks = responses_mask[:, 1:] 
        if self.args.use_last_reward:
            reward_score = self.get_last_reward_score(ref_values, responses_mask)
            
        else:
            reward_score = ref_values[:, :-1] * masks
        
        batch_size = reward_score.shape[0]
        rewards_with_kl_penalty, kl_penalty_all = [], []
        for i in range(batch_size):
            mask = masks[i]
            
            kl = actor_log_probs[i] - ref_log_probs[i]
            if self.args.kl_penalty_method == 'abs':
                kl = torch.abs(kl)
            elif self.args.kl_penalty_method == 'mse':
                kl = kl ** 2 * 0.5 
                
            kl_penalty = - self.args.kl_penalty_beta * kl 
            kl_penalty_all.append(kl_penalty)

            if self.args.use_last_reward:
                if self.args.reward_score_clip is not None:
                    reward_score[i] = torch.clamp(reward_score[i], -self.args.reward_score_clip, self.args.reward_score_clip)
                
                end_index = mask.nonzero()[-1].detach().item()
                kl_penalty[end_index] += reward_score[i]
            else:
                kl_penalty += reward_score[i]
                
            rewards_with_kl_penalty.append(kl_penalty)
        return torch.stack(rewards_with_kl_penalty), torch.stack(kl_penalty_all), reward_score 
    
    
    def get_advantages_and_returns(self, values, rewards, responses_mask):
        # Adopted from https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat
        lastgaelam = 0 
        advantages_reversed = []
        length = rewards.size()[-1]

        for t in reversed(range(length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]  
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam        
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        returns = advantages + values     
        return advantages.detach(), returns


    def get_responses_mask(self, sequences_mask, prompts_without_padding):
        batch_size = sequences_mask.shape[0]
        responses_mask = []
        for i in range(batch_size):
            prompt = prompts_without_padding[i]
            response_mask = torch.zeros_like(sequences_mask[i])
            response_mask[len(prompt):] = sequences_mask[i][len(prompt):]
            responses_mask.append(response_mask)
        return torch.stack(responses_mask)


    @torch.no_grad()
    def get_experience_data(self, prompts_ids):

        self.actor_model.eval()
        self.critic_model.eval()

        responses_ids = self.generate(prompts_ids, return_prompt=False)
        prompts_without_padding, responses_without_padding, sequences = self.process_sequences(prompts_ids, responses_ids)
        
        ### 不同进程填充      
        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            sequences["input_ids"] = self.accelerator.pad_across_processes(
                sequences["input_ids"], dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=pad_first
            )
            sequences["attention_mask"] = self.accelerator.pad_across_processes(
                sequences["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
        

        actor_logits, critic_values, _ = self.model(sequences)  
        with self.actor_model.disable_adapter():
            ## the same as sft model 
            ref_logits = self.actor_model(**sequences, return_dict=True).logits
            
        with self.critic_model.disable_adapter():
            ## the same as reward model 
            ## save current critic model v_head 
            v_head_stat_dict = self.critic_model.v_head.state_dict()
            setattr(self.critic_model, "critic_head_weight", v_head_stat_dict["summary.weight"])
            setattr(self.critic_model, "critic_head_bias", v_head_stat_dict["summary.bias"])
            ## change to reward model v_head
            self.critic_model.v_head.load_state_dict({"summary.weight": getattr(self.critic_model, "reward_head_weight"), "summary.bias": getattr(self.critic_model, "reward_head_bias")})

            ref_values = self.critic_model(**sequences)[-1]
            ## back to critic model v_head 
            self.critic_model.v_head.load_state_dict({"summary.weight": getattr(self.critic_model, "critic_head_weight"), "summary.bias": getattr(self.critic_model, "critic_head_bias")})
        
        
        actor_log_probs = self.get_log_probs(actor_logits[:, :-1, :], sequences["input_ids"][:, 1:]) 
        actor_ce = -self.masked_mean(actor_log_probs, sequences["attention_mask"][:, 1:], dim=-1)

        ref_log_probs = self.get_log_probs(ref_logits[:, :-1, :], sequences["input_ids"][:, 1:]) 
        ref_ce = -self.masked_mean(ref_log_probs, sequences["attention_mask"][:, 1:], dim=-1)

        responses_mask = self.get_responses_mask(sequences["attention_mask"], prompts_without_padding).to(self.device)
        
        rewards_with_kl_penalty, kl_penalty, reward_score = self.compute_rewards_with_kl_penalty(ref_values, actor_log_probs, ref_log_probs, responses_mask)

        critic_values = critic_values * responses_mask[:, 1:] 
        rewards_with_kl_penalty = rewards_with_kl_penalty * responses_mask[:, 1:]  
        advantages, returns = self.get_advantages_and_returns(critic_values, rewards_with_kl_penalty, responses_mask)

        self.actor_model.train()
        self.critic_model.train()
        
        return dict(
            prompts_ids=prompts_without_padding,
            responses_ids=responses_without_padding,
            responses_mask=responses_mask,
            sequences_ids=sequences["input_ids"],
            sequences_mask=sequences["attention_mask"],
            actor_log_probs=actor_log_probs,
            ref_log_probs=ref_log_probs,
            rewards_with_kl_penalty=rewards_with_kl_penalty,
            reward_score=reward_score,
            kl_penalty=kl_penalty,
            critic_values=critic_values,
            advantages=advantages,
            returns=returns,
            actor_ce=actor_ce,
            ref_ce=ref_ce,
        )


    def get_mini_dataset(self, data_buffer):

        mini_dataset = []
        batch_size = data_buffer[0]["exp"]["sequences_ids"].shape[0]
        for item in data_buffer:
            experience_data, batch_extra_data = item['exp'], item['extra']
            index = 0 
            while index < batch_size:
                dic = {}
                for k, v in experience_data.items():
                    if k in ["prompts_ids", "responses_ids"]:
                        dic[k] = v[index : index + self.args.per_device_mini_train_batch_size]
                    else:
                        dic[k] = v[index : index + self.args.per_device_mini_train_batch_size].to(self.device)
                        
                if batch_extra_data is not None:
                    for k, v in batch_extra_data.items():
                        dic[k] = v[index : index + self.args.per_device_mini_train_batch_size].to(self.device)
                
                mini_dataset.append(dic)
                index += self.args.per_device_mini_train_batch_size
 
        return mini_dataset 
        
        
    def actor_loss(self, actor_log_probs, mini_batch_actor_log_probs, advantages, mask):
        
        ratio = torch.exp((mini_batch_actor_log_probs - actor_log_probs) * mask)
        loss1 = -advantages * ratio
        loss2 = -advantages * torch.clamp(ratio, 1.0 - self.args.ratio_clip,
                                             1.0 + self.args.ratio_clip)

        loss = self.masked_mean(torch.max(loss1, loss2), mask)
        return loss, ratio 


    def critic_loss(self, critic_values, mini_batch_critic_values, returns, mask):
        
        critic_values_clip = torch.clamp(
            mini_batch_critic_values,
            critic_values - self.args.value_clip,
            critic_values + self.args.value_clip,
        )
        values_error = (mini_batch_critic_values - returns)**2 
        values_clip_error = (critic_values_clip - returns)**2 
        loss = 0.5 * self.masked_mean(torch.max(values_error, values_clip_error), mask)
        
        return loss, values_error 
    

    def get_state_dict(self, model):
        pretrained_model_state_dict = model.pretrained_model.state_dict()
        v_head_state_dict = model.v_head.state_dict()
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict 


    def save_checkpoint(self, model, output_dir, step, state_dict=None):
        
        output_dir = os.path.join(output_dir, f"checkpoint-{step}")
        logger.info(f"Saving model checkpoint to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            if hasattr(model, "v_head"):
                state_dict = self.get_state_dict(model)
            else:
                state_dict = model.state_dict()

        if isinstance(model, PreTrainedModel):  
            model.save_pretrained(output_dir, state_dict=state_dict)
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            adapter_state_dict = get_peft_model_state_dict(model, state_dict, adapter_name="default")

            if hasattr(model, "v_head"):
                ### add v_head (v_head not in modules_to_save)
                v_head_state_dict = model.v_head.state_dict()
                for k, v in v_head_state_dict.items():
                    adapter_state_dict[f"v_head.{k}"] = v 
            torch.save(adapter_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                
        try:
            model.peft_config.save_pretrained(output_dir)
        except AttributeError:
            model.peft_config['default'].save_pretrained(output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    
    def record_logs(self, batch):

        mask = batch["responses_mask"][:, 1:]
        prompt_lens = torch.tensor([len(prompt) for prompt in batch["prompts_ids"]], dtype=torch.float)
        response_lens = torch.tensor([len(response) for response in batch["responses_ids"]], dtype=torch.float)

        logs = dict()
        ## params
        logs["lr"] = self.optimizer.param_groups[0]['lr']
        
        ## loss
        logs["loss/actor"] = batch["actor_loss"]
        logs["loss/entropy"] = batch["entropy"]
        logs["loss/critic"] = batch["critic_loss"]
        logs["loss/extra"] = batch["extra_loss"]
        # logs["loss/all"] = batch["all_loss"]
        
        ## exp data
        if not self.args.use_last_reward:
            reward_score_mean = self.masked_mean(batch["reward_score"], mask)
            reward_score_var = self.masked_var(batch["reward_score"], mask)
        else:
            reward_score_mean = torch.mean(batch["reward_score"])
            reward_score_var = torch.var(batch["reward_score"])
            
        logs["exp_data/reward_score_mean"] = reward_score_mean
        logs["exp_data/reward_score_var"] = reward_score_var 
        
        logs["exp_data/kl_penalty_mean"] = self.masked_mean(batch["kl_penalty"], mask)
        logs["exp_data/kl_penalty_var"] = self.masked_var(batch["kl_penalty"], mask)

        logs["exp_data/rewards_with_kl_penalty_mean"] = self.masked_mean(batch["rewards_with_kl_penalty"], mask)
        logs["exp_data/rewards_with_kl_penalty_var"] = self.masked_var(batch["rewards_with_kl_penalty"], mask)
        
        logs["exp_data/actor_perplexity"] = math.exp(torch.mean(batch["actor_ce"]))
        logs["exp_data/ref_perplexity"] = math.exp(torch.mean(batch["ref_ce"]))
        
        ## actor
        logs["actor/advantages_mean"] = self.masked_mean(batch["advantages"], mask)
        logs["actor/advantages_var"] = self.masked_var(batch["advantages"], mask)
        
        logs["actor/ratio_mean"] = self.masked_mean(batch["ratio"], mask)
        logs["actor/ratio_var"] = self.masked_var(batch["ratio"], mask)
        
        ## critic
        logs["critic/returns_mean"] = self.masked_mean(batch["returns"], mask)
        logs["critic/returns_var"] = self.masked_var(batch["returns"], mask)

        logs["critic/values_error_mean"] = self.masked_mean(batch["values_error"], mask)
        logs["critic/values_error_var"] = self.masked_var(batch["values_error"], mask)
        
        ## length
        logs["length/prompts_length_mean"] = torch.mean(prompt_lens)
        logs["length/prompts_length_var"] = torch.var(prompt_lens)
        
        logs["length/responses_length_mean"] = torch.mean(response_lens)
        logs["length/responses_length_var"] = torch.var(response_lens)
        
        return logs


    def print_logs(self, all_logs, update_steps):

        all_logs_merged = {}
        for key in all_logs[0]:
            all_logs_merged[key] = torch.mean(torch.tensor([log[key] for log in all_logs])).to(self.device)
        
        if self.is_distributed:
            logs = {}
            torch.distributed.barrier()
            for k, v in all_logs_merged.items():
                if not isinstance(v, torch.Tensor):
                    warnings.warn(f"the log of {k} need to be tensors")
                    continue
                torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
                v /= self.accelerator.num_processes
                logs[k] = v 
            all_logs_merged = copy.deepcopy(logs) 
        
        
        if self.accelerator.is_main_process:
            logs = {}
            for k, v in all_logs_merged.items():
                logs[k] = v.cpu().numpy().item()
            self.accelerator.log(logs, step=int(update_steps))

            if update_steps > 0 and update_steps % self.args.logging_steps == 0:
                actor_loss, critic_loss, extra_loss = logs["loss/actor"], logs["loss/critic"], logs["loss/extra"]
                rewards_with_kl_penalty_mean = logs["exp_data/rewards_with_kl_penalty_mean"]
                lr = logs["lr"]
                print(f'update_steps:{update_steps}|lr:{lr}|actor_loss:{actor_loss}, critic_loss:{critic_loss}, extra_loss:{extra_loss}, rewards_with_kl_penalty_mean:{rewards_with_kl_penalty_mean}')
    
    
    def train_step(self, batch_mini_data, extra_inputs, step):
        
        extra_loss_weight_warmup = self.args.extra_loss_weight
        if self.args.extra_warmup_steps_ratio is not None:
            extra_warmup_steps = int(self.args.extra_warmup_steps_ratio * self.max_steps)
        ## get extra_loss_weight 
        if self.args.extra_warmup_steps_ratio is not None:
            if step < extra_warmup_steps:
                extra_loss_weight_warmup = step / extra_warmup_steps * self.args.extra_loss_weight
            else:
                extra_loss_weight_warmup = extra_loss_weight_warmup ** 1.001 


        responses_mask = batch_mini_data["responses_mask"]
        sequences = {"input_ids": batch_mini_data["sequences_ids"], "attention_mask": batch_mini_data["sequences_mask"]}

        with self.accelerator.accumulate(self.model):
            mini_batch_actor_logits, mini_batch_critic_values, extra_loss = self.model(sequences, extra_inputs)

        mini_batch_actor_log_probs = self.get_log_probs(mini_batch_actor_logits[:, :-1, :], batch_mini_data["sequences_ids"][:, 1:]) 
        entropy = self.get_entropy(mini_batch_actor_logits[:, :-1, :], responses_mask[:, 1:])
        
        actor_loss, ratio = self.actor_loss(batch_mini_data["actor_log_probs"], mini_batch_actor_log_probs, batch_mini_data["advantages"], responses_mask[:, 1:])
        
        
        critic_loss, values_error = self.critic_loss(batch_mini_data["critic_values"], mini_batch_critic_values, batch_mini_data["returns"], responses_mask[:, 1:])
        
        if extra_inputs is not None:
            loss = self.args.actor_loss_weight * actor_loss + self.args.entropy_beta * entropy + self.args.critic_loss_weight * critic_loss + extra_loss_weight_warmup * extra_loss
        else:
            loss = self.args.actor_loss_weight * actor_loss + self.args.entropy_beta * entropy + self.args.critic_loss_weight * critic_loss
        
        self.accelerator.backward(loss)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()

   
        return dict(
            all_loss=loss.detach(),
            actor_loss=actor_loss.detach(),
            critic_loss=critic_loss.detach(),
            extra_loss=extra_loss.detach() if extra_inputs is not None else 0.0,
            entropy=entropy.detach(),
            ratio=ratio.detach(),
            values_error=values_error.detach(),
            
        )
        
        
    def train(self):

        total_train_batch_size = (
            self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        )
        num_examples = self.num_examples(self.dataloader)
        if self.extra_train_dataloader is not None:
            extra_data_num_examples = self.num_examples(self.extra_train_dataloader)
        else:
            extra_data_num_examples = 0 
        
        if self.args.max_steps > 0:
            self.num_train_epochs = self.args.max_steps // self.num_update_steps_per_epoch + int(
                self.args.max_steps % self.num_update_steps_per_epoch > 0
            )
            self.max_steps = self.max_update_steps * self.args.gradient_accumulation_steps 
        else:
            self.num_train_epochs = math.ceil(self.args.num_train_epochs)
            self.max_steps = self.max_update_steps * self.args.gradient_accumulation_steps 

        if self.is_world_process_zero():
            # Train!
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}, Extra task examples = {extra_data_num_examples}")
            logger.info(f"  Num Epochs = {self.num_train_epochs:,}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total steps = {self.max_steps}, Total optimization steps = {self.max_update_steps}")


        progress_bar = tqdm(total=self.max_steps, disable=not self.is_world_process_zero())
        step = 0 
        data_buffer = list()
        all_logs = list()

        for epoch in range(int(self.num_train_epochs)):
            if self.extra_train_dataloader is None:
                self.extra_train_dataloader = [None] * len(self.dataloader)
            
            for i, (batch_data, batch_extra_data) in enumerate(zip(self.dataloader, self.extra_train_dataloader)):
                if i >= self.max_dataloader_iters:
                    break 

                prompts_ids = batch_data["input_ids"]
                experience_data = self.get_experience_data(prompts_ids)
                
                self.accelerator.wait_for_everyone()
                data_buffer.append({'exp': experience_data, 'extra': batch_extra_data})
                if len(data_buffer) == self.args.mini_data_buffer_nums:
                    mini_dataset = self.get_mini_dataset(data_buffer)
                    random.shuffle(mini_dataset) 
                    data_buffer.clear()

                    self.actor_model.train()
                    self.critic_model.train()
                    
                    for ppo_epoch in range(self.args.ppo_epochs):

                        for j, batch_mini_data in enumerate(mini_dataset):
                            step += 1 

                            if batch_extra_data is not None:
                                extra_inputs = {"input_ids": batch_mini_data["input_ids"], "labels": batch_mini_data["labels"]}
                            else:
                                extra_inputs = None 
                        
                        
                            result = self.train_step(batch_mini_data, extra_inputs, step)
                            batch_mini_data.update(result)
                            
                            progress_bar.update(1)

                            logs = self.record_logs(batch_mini_data)
                            all_logs.append(logs)
                            
                            update_steps = step / self.args.gradient_accumulation_steps
                            
                            if step > 0 and step % self.args.gradient_accumulation_steps == 0:
                                self.print_logs(all_logs, update_steps) 
                                all_logs.clear()
                            if update_steps > 0 and (update_steps % self.args.save_steps) == 0:
                                if self.is_world_process_zero():
                                    self.save_checkpoint(self.actor_model, self.args.output_dir, int(update_steps))
                                    self.save_checkpoint(self.critic_model, self.args.critic_output_dir, int(update_steps))

                        random.shuffle(mini_dataset) 
                        torch.cuda.empty_cache()

        progress_bar.close()
        self.accelerator.end_training()