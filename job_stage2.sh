#!/bin/bash

master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
export PYTHONPATH="${PYTHONPATH}:"


FINETUNE_NAME="mhealth_stage2"

torchrun --nproc_per_node=2 --master_port=$master_port sensorllm/train/train_mem.py   \
--model_name_or_path '' \
--pt_encoder_backbone_ckpt ''   \
--model_type "SequenceClassification" \
--num_labels 12  \
--use_weighted_loss True  \
--tokenize_method 'StanNormalizeUniformBins'    \
--dataset "mhealth" \
--data_path ''    \
--qa_path ''     \
--eval_data_path ''   \
--eval_qa_path ''    \
--preprocess_type "smry" \
--output_dir sensorllm/outputs/SensorLLM_train_stage2/${FINETUNE_NAME}    \
--model_max_length 4096    \
--num_train_epochs 8    \
--per_device_train_batch_size 4    \
--gradient_accumulation_steps 8    \
--per_device_eval_batch_size 4    \
--evaluation_strategy "steps"    \
--save_strategy "steps"    \
--save_steps 50    \
--eval_steps 50    \
--save_total_limit 1    \
--load_best_model_at_end True    \
--learning_rate 2e-3    \
--weight_decay 0.0    \
--warmup_ratio 0.03    \
--lr_scheduler_type "cosine"    \
--logging_steps 1    \
--bf16 True      \
--fix_llm True  \
--fix_cls_head False  \
--fix_ts_encoder True    \
--gradient_checkpointing True    \
--metric_for_best_model  "f1_macro" \
--greater_is_better True  \
--only_stage2 False	\
--stage_2 True  \
--shuffle True

