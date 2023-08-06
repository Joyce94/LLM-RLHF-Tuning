
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
    