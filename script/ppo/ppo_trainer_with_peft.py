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
    
    def forward(self, sequences, pretrain_inputs=None):
        actor_logits = self.actor_model(**sequences, return_dict=True).logits
        critic_values = self.critic_model(**sequences)[-1][:,:-1]
        if pretrain_inputs is not None:
            pretrain_loss = self.actor_model(**pretrain_inputs, return_dict=True).loss
        else:
            pretrain_loss = 0.0  
        return actor_logits, critic_values, pretrain_loss
    
    
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
        pretrain_train_dataset = None,
        pretrain_data_collator = None, 
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
        
        if pretrain_train_dataset is not None:
            self.pretrain_train_dataloader = DataLoader(
                pretrain_train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=pretrain_data_collator,
                num_workers=self.args.dataloader_num_workers,
                shuffle=True,
            )
            self.pretrain_train_dataloader = self.accelerator.prepare(self.pretrain_train_dataloader)
        else:
            self.pretrain_train_dataloader = None 
        
        self.tokenizer = tokenizer
        
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"
        if self.is_distributed:
            self.device = self.accelerator.device
        else:
            self.device = torch.device("cuda:0")
        
        # self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        # if self.is_deepspeed_enabled:
        #     if getattr(self.args, "hf_deepspeed_config", None) is None:
        #         from transformers.deepspeed import HfTrainerDeepSpeedConfig
        #         ds_plugin = self.accelerator.state.deepspeed_plugin

        #         ds_plugin.hf_ds_config = HfTrainerDeepSpeedConfig(ds_plugin.hf_ds_config.config)
        #         ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config
        #         ds_plugin.hf_ds_config.trainer_config_process(self.args)


        self.model = PPOModel(actor_model, critic_model)

        ## create optimizer and scheduler 
        self.optimizer, self.lr_scheduler = optimizers
        if self.optimizer is None:
            self.optimizer = self.create_optimizer(self.actor_model, self.critic_model, self.args.actor_lr, self.args.critic_lr, self.args.actor_weight_decay, self.args.critic_weight_decay)
        
        ## get max_update_steps for lr_scheduler 
        if self.pretrain_train_dataloader is None:
            self.max_dataloader_iters = len(self.dataloader)
        else:
            self.max_dataloader_iters = min(len(self.dataloader), len(self.pretrain_train_dataloader))
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
    
    
    def get_probs(self, logits, labels):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)  
        probs_labels = probs.gather(dim=-1, index=labels.unsqueeze(-1))
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)) 
        return probs, probs_labels.squeeze(-1), log_probs, log_probs_labels.squeeze(-1)


    def compute_rewards_with_kl_penalty(self, reward_score, actor_log_probs, ref_log_probs, responses_mask):
        
        batch_size = reward_score.shape[0]
        rewards_with_kl_penalty, r_kl_penalty = [], []
        for i in range(batch_size):
            kl_penalty = -self.args.kl_penalty_beta * (actor_log_probs[i] - ref_log_probs[i])
            r_kl_penalty.append(kl_penalty)
            
            end_index = responses_mask[i][1:].nonzero()[-1].detach().item()
            if self.args.reward_score_clip is not None:
                reward_score[i] = torch.clamp(reward_score[i], -self.args.reward_score_clip, self.args.reward_score_clip)
            
            kl_penalty[end_index] += reward_score[i]
            rewards_with_kl_penalty.append(kl_penalty)
        return torch.stack(rewards_with_kl_penalty), torch.stack(r_kl_penalty)
    
    
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
        actor_probs, actor_probs_labels, actor_log_probs, actor_log_probs_labels = self.get_probs(actor_logits[:, :-1, :], sequences["input_ids"][:, 1:])        
        
        with self.actor_model.disable_adapter():
            ## the same as sft model 
            ref_logits = self.actor_model(**sequences, return_dict=True).logits
            _, _, _, ref_log_probs_labels = self.get_probs(ref_logits[:, :-1, :], sequences["input_ids"][:, 1:]) 
            
        with self.critic_model.disable_adapter():
            ## the same as reward model 

            v_head_stat_dict = self.critic_model.v_head.state_dict()
            setattr(self.critic_model, "critic_head_weight", v_head_stat_dict["summary.weight"])
            setattr(self.critic_model, "critic_head_bias", v_head_stat_dict["summary.bias"])

            self.critic_model.v_head.load_state_dict({"summary.weight": getattr(self.critic_model, "reward_head_weight"), "summary.bias": getattr(self.critic_model, "reward_head_bias")})

            ref_values = self.critic_model(**sequences)[-1]
            self.critic_model.v_head.load_state_dict({"summary.weight": getattr(self.critic_model, "critic_head_weight"), "summary.bias": getattr(self.critic_model, "critic_head_bias")})
        

        responses_mask = self.get_responses_mask(sequences["attention_mask"], prompts_without_padding).to(self.device)
        
        reward_score = self.get_last_reward_score(ref_values, responses_mask)    
        rewards_with_kl_penalty, r_kl_penalty = self.compute_rewards_with_kl_penalty(reward_score, actor_log_probs_labels, ref_log_probs_labels, responses_mask)

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
            ref_log_probs_labels=ref_log_probs_labels,
            actor_log_probs=actor_log_probs,
            actor_log_probs_labels=actor_log_probs_labels,
            actor_probs_labels=actor_probs_labels,
            rewards_with_kl_penalty=rewards_with_kl_penalty,
            reward_score=reward_score,
            r_kl_penalty=r_kl_penalty,
            critic_values=critic_values,
            advantages=advantages,
            returns=returns
        )

        
    def get_mini_dataset(self, experience_data, batch_pretrain_data=None):

        if self.args.mini_data_shuffle:
            item = list(experience_data.items())
            random.shuffle(item)
            experience_data = dict(item)

        mini_dataset = []
        index = 0 
        batch_size = experience_data["sequences_ids"].shape[0]

        while index < batch_size:
            dic = {}
            for k, v in experience_data.items():
                if k in ["prompts_ids", "responses_ids"]:
                    dic[k] = v[index : index + self.args.per_device_mini_train_batch_size]
                else:
                    dic[k] = v[index : index + self.args.per_device_mini_train_batch_size].to(self.device)
                    
            if batch_pretrain_data is not None:
                for k, v in batch_pretrain_data.items():
                    dic[k] = v[index : index + self.args.per_device_mini_train_batch_size].to(self.device)
            mini_dataset.append(dic)
            index += self.args.per_device_mini_train_batch_size
        
        return mini_dataset 
        
        
    def actor_loss(self, actor_log_probs_labels, actor_log_probs, mini_batch_actor_logits, sequences_ids, advantages, mask):
        
        mini_batch_actor_probs, mini_batch_actor_probs_labels, mini_batch_actor_log_probs, mini_batch_actor_log_probs_labels= self.get_probs(mini_batch_actor_logits[:, :-1, :], sequences_ids[:, 1:]) 

        ratio = torch.exp((mini_batch_actor_log_probs_labels - actor_log_probs_labels) * mask)
        loss1 = advantages * ratio
        loss2 = advantages * torch.clamp(ratio, 1.0 - self.args.ratio_clip,
                                             1.0 + self.args.ratio_clip)

        entropy = -torch.sum(mini_batch_actor_probs * mini_batch_actor_log_probs, dim=-1) * mask
        kl_loss = torch.sum(mini_batch_actor_probs * (mini_batch_actor_log_probs - actor_log_probs), dim=-1) * mask 
        loss = -torch.min(loss1, loss2) - self.args.entropy_beta * entropy + self.args.kl_loss_alpha * kl_loss

        loss = self.masked_mean(loss, mask)
        return loss, ratio, entropy, kl_loss 


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
        logs["loss/critic"] = batch["critic_loss"]
        logs["loss/pretrain"] = batch["pretrain_loss"]
        logs["loss/all"] = batch["all_loss"]
        
        ## experience_data
        logs["exp_data/rewards_with_kl_penalty_mean"] = self.masked_mean(batch["rewards_with_kl_penalty"], mask)
        logs["exp_data/rewards_with_kl_penalty_var"] = self.masked_var(batch["rewards_with_kl_penalty"], mask)
        
        logs["exp_data/reward_score_mean"] = torch.mean(batch["reward_score"])
        logs["exp_data/reward_score_var"] = torch.var(batch["reward_score"])
        
        logs["exp_data/r_kl_penalty_mean"] = self.masked_mean(batch["r_kl_penalty"], mask)
        logs["exp_data/r_kl_penalty_var"] = self.masked_var(batch["r_kl_penalty"], mask)
        
        logs["exp_data/advantages_mean"] = self.masked_mean(batch["advantages"], mask)
        logs["exp_data/advantages_var"] = self.masked_var(batch["advantages"], mask)
        
        logs["exp_data/returns_mean"] = self.masked_mean(batch["returns"], mask)
        logs["exp_data/returns_var"] = self.masked_var(batch["returns"], mask)
        
        ## actor
        logs["actor/ratio_mean"] = self.masked_mean(batch["ratio"], mask)
        logs["actor/ratio_var"] = self.masked_var(batch["ratio"], mask)
        
        logs["actor/entropy_mean"] = self.masked_mean(batch["entropy"], mask)
        logs["actor/entropy_var"] = self.masked_var(batch["entropy"], mask)
        
        logs["actor/kl_loss_mean"] = self.masked_mean(batch["kl_loss"], mask)
        logs["actor/kl_loss_var"] = self.masked_var(batch["kl_loss"], mask)
        
        ## critic
        logs["critic/values_error_mean"] = self.masked_mean(batch["values_error"], mask)
        logs["critic/values_error_var"] = self.masked_var(batch["values_error"], mask)
        
        ## length
        logs["length/prompts_length_mean"] = torch.mean(prompt_lens)
        logs["length/prompts_length_var"] = torch.var(prompt_lens)
        
        logs["length/responses_length_mean"] = torch.mean(response_lens)
        logs["length/responses_length_var"] = torch.var(response_lens)
        
        return logs


    def print_logs(self, all_logs, step):
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
            self.accelerator.log(logs, step=step)

            update_steps = step / self.args.gradient_accumulation_steps
            if update_steps > 0 and update_steps % self.args.logging_steps == 0:
                actor_loss, critic_loss, pretrain_loss = logs["loss/actor"], logs["loss/critic"], logs["loss/pretrain"]
                rewards_with_kl_penalty_mean = logs["exp_data/rewards_with_kl_penalty_mean"]
                lr = logs["lr"]
                print(f'update_steps:{update_steps}|lr:{lr}|actor_loss:{actor_loss}, critic_loss:{critic_loss}, pretrain_loss:{pretrain_loss}, rewards_with_kl_penalty_mean:{rewards_with_kl_penalty_mean}')
    
    
    def train(self):

        total_train_batch_size = (
            self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
        )
        num_examples = self.num_examples(self.dataloader)
        if self.pretrain_train_dataloader is not None:
            pretrain_data_num_examples = self.num_examples(self.pretrain_train_dataloader)
        else:
            pretrain_data_num_examples = 0 
        
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
            logger.info(f"  Num examples = {num_examples}, Pretraining task examples = {pretrain_data_num_examples}")
            logger.info(f"  Num Epochs = {self.num_train_epochs:,}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total steps = {self.max_steps}, Total optimization steps = {self.max_update_steps}")


        progress_bar = tqdm(total=self.max_steps, disable=not self.is_world_process_zero())
        step = 0 

        pretrain_loss_weight_warmup = self.args.pretrain_loss_weight
            
        for epoch in range(int(self.num_train_epochs)):
            if self.pretrain_train_dataloader is None:
                self.pretrain_train_dataloader = [None] * len(self.dataloader)
            
            for i, (batch_data, batch_pretrain_data) in enumerate(zip(self.dataloader, self.pretrain_train_dataloader)):
                if i >= self.max_dataloader_iters:
                    break 

                prompts_ids = batch_data["input_ids"]
                experience_data = self.get_experience_data(prompts_ids)
                
                self.accelerator.wait_for_everyone()
                mini_dataset = self.get_mini_dataset(experience_data, batch_pretrain_data)

                self.actor_model.train()
                self.critic_model.train()
                
                for ppo_epoch in range(self.args.ppo_epochs):
                    all_logs = []
                    for j, batch_mini_data in enumerate(mini_dataset):
                        step += 1 
                        ## calc pretrain_loss_weight 
                        if self.args.pretrain_warmup_steps is not None:
                            if step < self.args.pretrain_warmup_steps:
                                pretrain_loss_weight_warmup = step / self.args.pretrain_warmup_steps * self.args.pretrain_loss_weight
                            else:
                                pretrain_loss_weight_warmup = pretrain_loss_weight_warmup ** 1.001 

                        responses_mask = batch_mini_data["responses_mask"]
                        sequences = {"input_ids": batch_mini_data["sequences_ids"], "attention_mask": batch_mini_data["sequences_mask"]}

                        if batch_pretrain_data is not None:
                            pretrain_inputs = {"input_ids": batch_mini_data["input_ids"], "labels": batch_mini_data["labels"]}
                        else:
                            pretrain_inputs = None 
                            
                        with self.accelerator.accumulate(self.model):
                            
                            mini_batch_actor_logits, mini_batch_critic_values, pretrain_loss = self.model(sequences, pretrain_inputs)

                            actor_loss, ratio, entropy, kl_loss = self.actor_loss(batch_mini_data["actor_log_probs_labels"], batch_mini_data["actor_log_probs"], mini_batch_actor_logits, batch_mini_data["sequences_ids"], batch_mini_data["advantages"], responses_mask[:, 1:])
                            
                            critic_loss, values_error = self.critic_loss(batch_mini_data["critic_values"], mini_batch_critic_values, batch_mini_data["returns"], responses_mask[:, 1:])
                            
                            if pretrain_inputs is not None:
                                loss = self.args.actor_loss_weight * actor_loss + self.args.critic_loss_weight * critic_loss + pretrain_loss_weight_warmup * pretrain_loss
                            else:
                                loss = self.args.actor_loss_weight * actor_loss + self.args.critic_loss_weight * critic_loss
                            self.accelerator.backward(loss)
                            
                            self.optimizer.step()
                            if self.lr_scheduler is not None:
                                self.lr_scheduler.step()
                            self.optimizer.zero_grad()
                        
                        progress_bar.update(1)

                        batch_mini_data["actor_loss"] = actor_loss.detach()
                        batch_mini_data["critic_loss"] = critic_loss.detach()
                        
                        batch_mini_data["all_loss"] = loss.detach()
                        batch_mini_data["ratio"] = ratio.detach()
                        batch_mini_data["entropy"] = entropy.detach()
                        batch_mini_data["kl_loss"] = kl_loss.detach()
                        batch_mini_data["values_error"] = values_error.detach()
                        
                        if pretrain_inputs is not None:
                            batch_mini_data["pretrain_loss"] = pretrain_loss.detach()
                        else:
                            batch_mini_data["pretrain_loss"] = 0.0
                        
                        logs = self.record_logs(batch_mini_data)
                        all_logs.append(logs)
                        
                        update_steps = step / self.args.gradient_accumulation_steps
                        if update_steps > 0 and (update_steps % self.args.save_steps) == 0:
                            if self.is_world_process_zero():
                                self.save_checkpoint(self.actor_model, self.args.output_dir, int(update_steps))
                                self.save_checkpoint(self.critic_model, self.args.critic_output_dir, int(update_steps))

                    
                    self.print_logs(all_logs, step)                
                    random.shuffle(mini_dataset) 
                    torch.cuda.empty_cache()

        progress_bar.close()
        self.accelerator.end_training()

