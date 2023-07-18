
deepspeed_config_file=ds.json
sft_model_path=chinese_llama_path
rm_model_path=rm_model_path
dataset_dir=sft_data
pretrain_dataset_dir=pt_data
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"    # v_head

critic_output_dir=output_dir_rlhf_critic
output_dir=output_dir_rlhf_actor
actor_modules_to_save="embed_tokens,lm_head,v_head"
critic_modules_to_save="embed_tokens,lm_head,v_head"

torchrun --nnodes 1 --nproc_per_node 1 run_rl_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_type llama \
    --sft_model_path ${sft_model_path} \
    --rm_model_path ${rm_model_path} \
    --dataset_dir ${dataset_dir} \
    --split_ratio 0.01 \
    --pretrain_dataset_dir ${pretrain_dataset_dir} \
    --per_device_train_batch_size 1 \
    --per_device_mini_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --seed 512 \
    --fp16 \
    --actor_lr 2e-4 \
    --critic_lr 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --save_strategy steps \
    --save_total_limit 1 \
    --logging_steps 10 \
    --save_steps 10 \
    --preprocessing_num_workers 16 \
    --block_size 512 \
    --output_dir ${output_dir} \
    --critic_output_dir ${critic_output_dir} \
    --overwrite_output_dir \
    --logging_first_step True \
    --actor_lora_rank 8 \
    --actor_lora_alpha 32 \
    --actor_lora_target ${lora_trainable} \
    --actor_lora_dropout 0.05 \
    --actor_modules_to_save ${actor_modules_to_save} \
    --critic_lora_rank 8 \
    --critic_lora_alpha 32 \
    --critic_lora_target ${lora_trainable} \
    --critic_lora_dropout 0.05 \
    --critic_modules_to_save ${critic_modules_to_save} \
    --max_prompt_length 256 \
    --max_response_length 256 \
    --num_train_rl_epochs 1 \
    --gamma 1 \
    --lam 0.95 \
    --kl_penalty_beta 0.02 \
    --reward_clip 10 \
    --value_clip 0.2 \
    --ratio_clip 0.2 \
    --pt_weight 1 \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    