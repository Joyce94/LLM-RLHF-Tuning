
pretrained_model=chinese_llama_path
dataset_dir=/root/LLM-RLHF-Tuning/sft_data
data_cache_dir=/root/LLM-RLHF-Tuning/sft_data/cache/data
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