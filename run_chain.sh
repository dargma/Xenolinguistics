#!/usr/bin/env bash
set -e
SMOKE_DIR=outputs/fastdllm_smoke
FULL_DIR=outputs/fastdllm_v2_100k_lora64
mkdir -p "$SMOKE_DIR" "$FULL_DIR"

echo "[$(date)] SMOKE start"
python3 -u train_fastdllm.py \
  --train_file data/train_1k.jsonl \
  --output_dir "$SMOKE_DIR" \
  --lr 5e-5 --epochs 5 --batch_size 1 --grad_accum 8 --max_len 512 \
  --lora_rank 64 --lora_alpha 128 \
  --save_steps 999999 > "$SMOKE_DIR/smoke.log" 2>&1
echo "[$(date)] SMOKE exit=$?"

N=$(grep "'loss':" "$SMOKE_DIR/smoke.log" | grep -vE "'loss': '(0|nan)'" | wc -l)
echo "FINITE_LOSS=$N"
if [ "$N" -lt 5 ]; then
  echo "SMOKE_FAIL — abort"
  exit 2
fi

echo "[$(date)] FULL start"
python3 -u train_fastdllm.py \
  --train_file data/train_100k.jsonl \
  --output_dir "$FULL_DIR" \
  --lr 5e-5 --epochs 1 --batch_size 1 --grad_accum 8 --max_len 512 \
  --lora_rank 64 --lora_alpha 128 \
  --save_steps 2000 > "$FULL_DIR/train.log" 2>&1
echo "[$(date)] FULL exit=$?"
