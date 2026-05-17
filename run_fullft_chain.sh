#!/usr/bin/env bash
set -e
# Wait until r=256 chain (PID 36479) exits
while kill -0 36479 2>/dev/null; do sleep 60; done
echo "[$(date)] r=256 chain PID 36479 exited. Starting Full FT phase."

echo "[$(date)] Phase 3: Fast-dLLM v2 Full FT 100k"
mkdir -p outputs/fastdllm_v2_100k_fullft
python3 -u train_fastdllm.py \
  --train_file data/train_100k.jsonl \
  --output_dir outputs/fastdllm_v2_100k_fullft \
  --lr 2e-5 --epochs 1 --batch_size 1 --grad_accum 8 --max_len 512 \
  --lora_rank 0 \
  --save_steps 2000 > outputs/fastdllm_v2_100k_fullft/train.log 2>&1
python3 -u eval_fastdllm.py --ckpt outputs/fastdllm_v2_100k_fullft/final \
  --mode free --out outputs/fastdllm_v2_100k_fullft/eval_free.json \
  > outputs/fastdllm_v2_100k_fullft/eval_free.log 2>&1
python3 -u eval_fastdllm.py --ckpt outputs/fastdllm_v2_100k_fullft/final \
  --mode gt_length --out outputs/fastdllm_v2_100k_fullft/eval_gtlen.json \
  > outputs/fastdllm_v2_100k_fullft/eval_gtlen.log 2>&1
echo "[$(date)] FastDLLM Full FT done"

echo "[$(date)] Phase 4: Qwen2.5-7B Full FT 100k"
mkdir -p outputs/qwen_100k_fullft
python3 -u train_qwen_v4.py \
  --train_file data/train_100k.jsonl --val_file data/val_1k.jsonl \
  --output_dir outputs/qwen_100k_fullft \
  --lora_rank 0 \
  --epochs 1 --lr 2e-5 > outputs/qwen_100k_fullft/train.log 2>&1
python3 -u eval_qwen.py --adapter outputs/qwen_100k_fullft/final \
  --out outputs/qwen_100k_fullft/eval.json > outputs/qwen_100k_fullft/eval.log 2>&1
echo "[$(date)] Qwen Full FT done. ALL DONE."
