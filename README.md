##### This project implemented a complete RLHF training process from scratch, mainly for personal learning.

### step 1: supervised finetuning
```
pretrained_model=chinese_llama_path
dataset_dir=/root/Chinese-LLaMA-Tuning/sft_data
data_cache_dir=/root/Chinese-LLaMA-Tuning/sft_data/cache/data
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
output_dir=sft_model_path
peft_path=sft_lora_model
modules_to_save=None

torchrun --nnodes 1 --nproc_per_node 1 run_sft_with_peft.py \
    --model_type llama \
    --model_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --split_ratio 0.01 \
    --data_cache_dir ${data_cache_dir} \
    --block_size 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --do_eval \
    --seed 512 \
    --fp16 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --save_strategy steps \
    --evaluation_strategy steps \
    --save_total_limit 1 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_target ${lora_trainable} \
    --lora_dropout 0.05 \
    --modules_to_save ${modules_to_save} \
    --torch_dtype float16 \
    --report_to "wandb"
```

### step 2: reward model finetuning
```
pretrained_model=chinese_alpaca_path
dataset_dir=/root/Chinese-LLaMA-Tuning/rm_data
data_cache_dir=/root/Chinese-LLaMA-Tuning/rm_data/cache/data
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
output_dir=rm_lora_path

torchrun --nnodes 1 --nproc_per_node 1 run_rm_with_peft.py \
    --model_type llama \
    --model_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --split_ratio 0.01 \
    --data_cache_dir ${data_cache_dir} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --do_eval \
    --seed 512 \
    --fp16 \
    --num_train_epochs 1 \
    --max_length 1024 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --evaluation_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 1 \
    --eval_steps 10 \
    --save_steps 10 \
    --block_size 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --logging_first_step True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_target ${lora_trainable} \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --report_to "wandb"
    
```

### step 3: use Proximal Policy Optimization 
```

sft_model_path=chinese_alpaca_path
reward_lora_path=rm_lora_path
actor_peft_path=output_dir_rlhf_actor
critic_peft_path=output_dir_rlhf_critic

dataset_dir=/root/Chinese-LLaMA-Tuning/sft_data
pretrain_dataset_dir=/root/Chinese-LLaMA-Tuning/pt_data
actor_lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"    
critic_lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"   

actor_output_dir=output_dir_rlhf_actor
critic_output_dir=output_dir_rlhf_critic

accelerate launch --config_file default_config.yaml run_ppo_with_peft.py \
    --model_type llama \
    --sft_model_path ${sft_model_path} \
    --reward_lora_path ${reward_lora_path} \
    --dataset_dir ${dataset_dir} \
    --pretrain_dataset_dir ${pretrain_dataset_dir} \
    --per_device_train_batch_size 2 \
    --per_device_mini_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --num_train_epochs 1 \
    --seed 512 \
    --lr_scheduler_type cosine \
    --actor_lr 1e-4 \
    --critic_lr 1e-4 \
    --logging_steps 10 \
    --save_steps 10 \
    --dataloader_num_workers 16 \
    --output_dir ${actor_output_dir} \
    --critic_output_dir ${critic_output_dir} \
    --actor_lora_rank 8 \
    --actor_lora_alpha 32 \
    --actor_lora_target ${actor_lora_trainable} \
    --actor_lora_dropout 0.05 \
    --critic_lora_rank 8 \
    --critic_lora_alpha 32 \
    --critic_lora_target ${critic_lora_trainable} \
    --critic_lora_dropout 0.05 \
    --max_prompt_length 256 \
    --max_response_length 256 \
    --ppo_epochs 1 \
    --gamma 1 \
    --lam 0.95 \
    --kl_penalty_beta 0.02 \
    --reward_score_clip 10 \
    --value_clip 0.2 \
    --ratio_clip 0.2 \
    --actor_loss_weight 1 \
    --critic_loss_weight 2 \
    --pretrain_loss_weight 0.1 \
    --pretrain_warmup_steps 500 \
    --report_to "wandb" \
    --torch_dtype float16 \
    --fp16
```
