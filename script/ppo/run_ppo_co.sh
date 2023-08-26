sft_model_path=chinese_alpaca_path
reward_lora_path=rm_lora_path
peft_path=output_dir_rlhf

dataset_dir=/root/LLM-RLHF-Tuning/sft_data
extra_dataset_dir=/root/LLM-RLHF-Tuning/pt_data
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"    

output_dir=output_dir_rlhf

accelerate launch --config_file default_config.yaml run_ppo_with_peft.py \
    --model_type llama \
    --template "chinese_llama2_alpaca" \
    --use_co_model True \
    --sft_model_path ${sft_model_path} \
    --reward_lora_path ${reward_lora_path} \
    --dataset_dir ${dataset_dir} \
    --extra_dataset_dir ${extra_dataset_dir} \
    --per_device_train_batch_size 1 \
    --per_device_mini_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --num_train_epochs 1 \
    --seed 512 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --logging_steps 10 \
    --save_steps 10 \
    --dataloader_num_workers 16 \
    --block_size 256 \
    --max_prompt_length 256 \
    --max_response_length 256 \
    --output_dir ${output_dir} \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_target ${lora_trainable} \
    --lora_dropout 0.05 \
    --ppo_epochs 1 \
    --gamma 1 \
    --lam 0.95 \
    --kl_penalty_beta 0.02 \
    --kl_penalty_method "abs" \
    --value_clip 0.2 \
    --ratio_clip 0.2 \
    --actor_loss_weight 1 \
    --critic_loss_weight 1 \
    --extra_loss_weight 0.2 \
    --extra_warmup_steps_ratio 0.2 \
    --entropy_beta 1.0 \
    --use_advantage_norm \
    --report_to "wandb" \
    --torch_dtype float16 \
    --fp16