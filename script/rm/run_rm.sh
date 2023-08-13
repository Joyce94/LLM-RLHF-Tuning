
pretrained_model=chinese_alpaca_path
dataset_dir=/root/LLM-RLHF-Tuning/rm_data
data_cache_dir=/root/LLM-RLHF-Tuning/rm_data/cache/data
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
output_dir=rm_lora_path

torchrun --nnodes 1 --nproc_per_node 1 run_rm_with_peft.py \
    --model_type llama \
    --model_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --split_ratio 0.01 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 8 \
    --do_train \
    --do_eval \
    --seed 512 \
    --fp16 \
    --num_train_epochs 1 \
    --max_length 512 \
    --clm_loss_weight 1.0 \
    --use_last_reward \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --logging_strategy steps \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_strategy steps \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --output_dir ${output_dir} \
    --logging_first_step True \
    --lora_rank 128 \
    --lora_alpha 32 \
    --lora_target ${lora_trainable} \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --report_to "wandb"