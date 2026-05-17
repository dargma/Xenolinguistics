#!/usr/bin/env bash
set -e
echo "[$(date)] Phase 1: Fast-dLLM v2 LoRA r=256 100k"
mkdir -p outputs/fastdllm_v2_100k_lora256
python3 -u train_fastdllm.py \
  --train_file data/train_100k.jsonl \
  --output_dir outputs/fastdllm_v2_100k_lora256 \
  --lr 5e-5 --epochs 1 --batch_size 1 --grad_accum 8 --max_len 512 \
  --lora_rank 256 --lora_alpha 512 --lora_target all-linear \
  --save_steps 2000 > outputs/fastdllm_v2_100k_lora256/train.log 2>&1
echo "[$(date)] FastDLLM train done"
python3 -u eval_fastdllm.py --ckpt outputs/fastdllm_v2_100k_lora256/final \
  --mode free --out outputs/fastdllm_v2_100k_lora256/eval_free.json \
  > outputs/fastdllm_v2_100k_lora256/eval_free.log 2>&1
python3 -u eval_fastdllm.py --ckpt outputs/fastdllm_v2_100k_lora256/final \
  --mode gt_length --out outputs/fastdllm_v2_100k_lora256/eval_gtlen.json \
  > outputs/fastdllm_v2_100k_lora256/eval_gtlen.log 2>&1
echo "[$(date)] FastDLLM eval done"

echo "[$(date)] Phase 2: Qwen2.5-7B LoRA r=256 100k"
mkdir -p outputs/qwen_100k_lora256
python3 -u train_qwen_v4.py \
  --train_file data/train_100k.jsonl --val_file data/val_1k.jsonl \
  --output_dir outputs/qwen_100k_lora256 \
  --lora_rank 256 --lora_alpha 512 --target_modules all-linear \
  --epochs 1 --lr 2e-4 > outputs/qwen_100k_lora256/train.log 2>&1
echo "[$(date)] Qwen train done"
python3 -u eval_qwen.py --adapter outputs/qwen_100k_lora256/adapter \
  --out outputs/qwen_100k_lora256/eval.json > outputs/qwen_100k_lora256/eval.log 2>&1
echo "[$(date)] Qwen eval done. ALL DONE."
