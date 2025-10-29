#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:$(pwd)/../
export WORLD_SIZE=4
export TRITON_CACHE_DIR="~/triton-cache"  # Avoid NFS warning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

deepspeed --num_gpus 4 --module openrlhf.cli.train_sft \
   --max_len=2048 \
   --dataset=talzoomanzoo/math_compilation_gsm8k_contam_math_uncontam \
   --input_key=Question \
   --output_key=answer \
   --train_batch_size=64 \
   --micro_train_batch_size=1 \
   --pretrain=Qwen/Qwen2.5-Math-1.5B \
   --save_path=./checkpoint/compiled-full-gsm8k_contam_math_uncontam \
   --packing_samples \
   --flash_attn \
   --logging_steps=1 \
   --eval_steps=-1 \
   --zero_stage=2 \
   --max_epochs=10 \
   --learning_rate=1e-5 \
   --gradient_checkpointing \
   --bf16 \
   --use_wandb=True \
   --wandb_org=mjgwak \
   --wandb_project=benbench-contamination-poc \
   --wandb_run_name=compiled-full-gsm8k_contam_math_uncontam-$(date +%m%d%H%M)