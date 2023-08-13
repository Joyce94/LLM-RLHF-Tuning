
sft_model_path=chinese_alpaca_path
reward_lora_path=rm_lora_path
actor_peft_path=output_dir_rlhf_actor
critic_peft_path=output_dir_rlhf_critic

dataset_dir=/root/LLM-RLHF-Tuning/sft_data
pretrain_dataset_dir=/root/LLM-RLHF-Tuning/pt_data
actor_lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"    
critic_lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"   

actor_output_dir=output_dir_rlhf_actor
critic_output_dir=output_dir_rlhf_critic

accelerate launch --config_file default_config.yaml run_ppo_with_peft.py \
    --model_type llama \
    --sft_model_path ${sft_model_path} \
    --reward_lora_path ${reward_lora_path} \
    --dataset_dir ${dataset_dir} \
    --extra_dataset_dir ${extra_dataset_dir} \
    --per_device_train_batch_size 2 \
    --per_device_mini_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --num_train_epochs 1 \
    --seed 512 \
    --lr_scheduler_type cosine \
    --actor_lr 1e-4 \
    --critic_lr 1e-4 \
    --logging_steps 100 \
    --save_steps 100 \
    --dataloader_num_workers 16 \
    --block_size 256 \
    --max_prompt_length 256 \
    --max_response_length 512 \
    --output_dir ${actor_output_dir} \
    --critic_output_dir ${critic_output_dir} \
    --actor_lora_rank 64 \
    --actor_lora_alpha 32 \
    --actor_lora_target ${actor_lora_trainable} \
    --actor_lora_dropout 0.05 \
    --critic_lora_rank 64 \
    --critic_lora_alpha 32 \
    --critic_lora_target ${critic_lora_trainable} \
    --critic_lora_dropout 0.05 \
    --ppo_epochs 1 \
    --gamma 1 \
    --lam 0.95 \
    --kl_penalty_beta 0.02 \
    --use_last_reward \
    --reward_score_clip 10 \
    --value_clip 0.2 \
    --ratio_clip 0.2 \
    --actor_loss_weight 1 \
    --critic_loss_weight 1 \
    --extra_loss_weight 0.2 \
    --extra_warmup_steps_ratio 0.2 \
    --entropy_beta 0.0 \
    --kl_loss_alpha 0.0 \
    --report_to "wandb" \
    --torch_dtype float16 \
    --fp16