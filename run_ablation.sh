#!/bin/bash
# Dream v4 Ablation Experiments
# Usage: bash run_ablation.sh [exp_number]
# GPU가 비면 하나씩 실행

cd /content/drive/MyDrive/Xenolinguistics
TRAIN=train_dream_v4.py

# Exp-1: 공식 설정 완전 재현 (all-linear + lr2e-6 + cart + max128)
if [ "$1" = "1" ] || [ "$1" = "all" ]; then
echo "=== Exp-1: Official config (all-linear + lr2e-6 + cart) ==="
python3 $TRAIN \
  --experiment_name exp1_official_config \
  --target_modules all-linear \
  --lr 2e-6 \
  --time_reweighting cart \
  --max_length 128 \
  --train_file data/train_10k.jsonl \
  --val_file data/val_10k.jsonl \
  --output_dir outputs/dream_v4_exp1
fi

# Exp-2: all-linear만 (lr/rw는 기존 유지)
if [ "$1" = "2" ] || [ "$1" = "all" ]; then
echo "=== Exp-2: all-linear only ==="
python3 $TRAIN \
  --experiment_name exp2_all_linear_only \
  --target_modules all-linear \
  --lr 1e-5 \
  --time_reweighting original \
  --max_length 512 \
  --train_file data/train_10k.jsonl \
  --val_file data/val_10k.jsonl \
  --output_dir outputs/dream_v4_exp2
fi

# Exp-3: lr=2e-6만 (modules는 기존 유지)
if [ "$1" = "3" ] || [ "$1" = "all" ]; then
echo "=== Exp-3: lr=2e-6 only ==="
python3 $TRAIN \
  --experiment_name exp3_lr2e6_only \
  --target_modules q_proj,v_proj,k_proj,o_proj \
  --lr 2e-6 \
  --time_reweighting original \
  --max_length 512 \
  --train_file data/train_10k.jsonl \
  --val_file data/val_10k.jsonl \
  --output_dir outputs/dream_v4_exp3
fi

# Exp-4: cart만 (modules/lr은 기존 유지)
if [ "$1" = "4" ] || [ "$1" = "all" ]; then
echo "=== Exp-4: CART reweighting only ==="
python3 $TRAIN \
  --experiment_name exp4_cart_only \
  --target_modules q_proj,v_proj,k_proj,o_proj \
  --lr 1e-5 \
  --time_reweighting cart \
  --max_length 512 \
  --train_file data/train_10k.jsonl \
  --val_file data/val_10k.jsonl \
  --output_dir outputs/dream_v4_exp4
fi

# Exp-5: max_length=128 (Exp-1 기반)
# Exp-1과 동일하므로 별도 실행 불필요 (Exp-1이 이미 128)
# 대신 max_length=256으로 비교
if [ "$1" = "5" ] || [ "$1" = "all" ]; then
echo "=== Exp-5: max_length=256 (Exp-1 base) ==="
python3 $TRAIN \
  --experiment_name exp5_maxlen256 \
  --target_modules all-linear \
  --lr 2e-6 \
  --time_reweighting cart \
  --max_length 256 \
  --train_file data/train_10k.jsonl \
  --val_file data/val_10k.jsonl \
  --output_dir outputs/dream_v4_exp5
fi

# Exp-6: LoRA r=64 (Exp-1 기반)
if [ "$1" = "6" ] || [ "$1" = "all" ]; then
echo "=== Exp-6: LoRA r=64 (Exp-1 base) ==="
python3 $TRAIN \
  --experiment_name exp6_lora_r64 \
  --target_modules all-linear \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lr 2e-6 \
  --time_reweighting cart \
  --max_length 128 \
  --train_file data/train_10k.jsonl \
  --val_file data/val_10k.jsonl \
  --output_dir outputs/dream_v4_exp6
fi

# Exp-7: Filtered data (Exp-1 기반)
if [ "$1" = "7" ] || [ "$1" = "all" ]; then
echo "=== Exp-7: Filtered data (Exp-1 base) ==="
python3 $TRAIN \
  --experiment_name exp7_filtered \
  --target_modules all-linear \
  --lr 2e-6 \
  --time_reweighting cart \
  --max_length 128 \
  --train_file data/train_10k_filtered.jsonl \
  --val_file data/val_10k_filtered.jsonl \
  --output_dir outputs/dream_v4_exp7
fi

echo "=== Done ==="
