#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:$(pwd)/../
export WORLD_SIZE=4
export TRITON_CACHE_DIR="~/triton-cache"  # Avoid NFS warning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

deepspeed --num_gpus 4 --module openrlhf.cli.train_sft \
   --max_len=32768 \
   --dataset=talzoomanzoo/aime_train \
   --input_key=Question \
   --output_key=answer \
   --train_batch_size=4 \
   --micro_train_batch_size=1 \
   --pretrain=talzoomanzoo/aime-tuned \
   --save_path=./checkpoint/aime-full-tuned \
   --packing_samples \
   --flash_attn \
   --logging_steps=1 \
   --eval_steps=-1 \
   --zero_stage=2 \
   --max_epochs=3 \
   --learning_rate=1e-5 \
   --gradient_checkpointing \
   --bf16 \
   --use_wandb=True \
   --wandb_org=mjgwak \
   --wandb_project=benbench-contamination-check-reference\
   --wandb_run_name=aime-full-tuned-$(date +%m%d%H%M)