
deepspeed_config_file=ds_zero2_no_offload.json
pretrained_model=/root/autodl-tmp/llama-7b-hf-transformers-4.29
chinese_tokenizer_path=/root/LLaMA-Tuning/scripts/merged_tokenizer_hf
dataset_dir=/root/LLaMA-Tuning/sft_data
cache_dir=./cache/model
data_cache_dir=./cache/data
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
output_dir=/root/autodl-tmp/output_dir3
# peft_model=/root/LLaMA-Tuning/scripts/output_dir/pt_lora_model
modules_to_save="embed_tokens,lm_head"


torchrun --nnodes 1 --nproc_per_node 1 run_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --cache_dir ${cache_dir} \
    --data_cache_dir ${data_cache_dir} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --seed 512 \
    --fp16 \
    --max_steps 100 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 500 \
    --preprocessing_num_workers 8 \
    --block_size 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --logging_first_step True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_target ${lora_trainable} \
    --lora_dropout 0.05 \
    # --modules_to_save ${modules_to_save} \
    # --peft_path ${peft_model} 
    --torch_dtype float16 \
    
    