#!/bin/bash

master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
export PYTHONPATH="${PYTHONPATH}:"

FINETUNE_NAME="mhealth_stage1"

torchrun --nproc_per_node=2 --master_port=$master_port sensorllm/train/train_mem.py   \
--model_name_or_path ./llama/Meta-Llama-3-8B-Instruct   \
--pt_encoder_backbone_ckpt ./chronos/chronos-t5-large   \
--tokenize_method 'StanNormalizeUniformBins'   \
--dataset "mhealth"   \
--data_path ''   \
--eval_data_path ''   \
--qa_path ''   \
--eval_qa_path ''   \
--output_dir sensorllm/outputs/SensorLLM_train_stage1/${FINETUNE_NAME}   \
--model_max_length 4096   \
--num_train_epochs  8   \
--per_device_train_batch_size 4   \
--gradient_accumulation_steps 8   \
--per_device_eval_batch_size 4   \
--evaluation_strategy "steps"    \
--save_strategy "steps"   \
--save_steps 2000   \
--eval_steps 2000   \
--load_best_model_at_end True   \
--save_total_limit 1    \
--learning_rate 2e-3   \
--weight_decay 0.0   \
--warmup_ratio 0.03   \
--lr_scheduler_type "cosine"   \
--logging_steps 1   \
--bf16 True    \
--fix_llm True   \
--fix_ts_encoder True   \
--gradient_checkpointing True   \
--model_type CasualLM   \
--shuffle False
