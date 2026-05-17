# Xenolinguistics — AR LLM vs Diffusion LLM, en→fi

Apple-to-apple comparison of an **AR LLM** (`Qwen/Qwen2.5-7B-Instruct`)
and a **Diffusion LLM** (`Efficient-Large-Model/Fast_dLLM_v2_7B`) on
English → Finnish translation (OPUS-100). Same backbone family, same data.

| File | Purpose |
|---|---|
| `README.md` (this file) | Install + commands + results — single source. |
| [`CLAUDE.md`](CLAUDE.md) | Hard rules for AI assistants. |
| [`reports/REPORT_GUIDE.md`](reports/REPORT_GUIDE.md) | How to log a new run. |

---

## 1. Results (2026-05-17, 100k OPUS-100)

Test set = `data/test_1k.jsonl` first 100. Metric = `sacrebleu` defaults.
Each model → `outputs/<run>/eval_free.json` (and `eval_gt_length.json` when present).

| Model | Method | Train progress | chrF (free) | BLEU (free) | chrF (gt_length) | BLEU (gt_length) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `opus-mt-tc-big-en-fi` | NMT baseline | — | **56.91** | **37.45** | — | — |
| Qwen2.5-7B | Full FT, lr=2e-5 | 1 ep, 7500/10000 (75%, disk-full) | **47.72** | **27.65** | pending | pending |
| Fast-dLLM v2 7B | Full FT, lr=2e-5 | 1 ep, 10000/10000 (100%) | 40.86 | 17.22 | pending | pending |
| Fast-dLLM v2 7B | LoRA r=256, lr=5e-5 | 1 ep, 10000/10000 (100%) | 37.22 | 15.38 | pending | pending |

`free` = generate until first EOS, 256 cap. `gt_length` = oracle target-len cut,
EOS ignored. See §6 for exact semantics.

Same backbone (Qwen2.5-7B family) → differences trace to the generation
paradigm (AR vs block masked diffusion), not pretraining data.

**Status**: `free` eval done for all rows. `gt_length` re-eval under unified
semantics interrupted — pending. Qwen LoRA r=256 100k pending (failed twice
on disk).

---

## 2. Environment

| Item | Verified | Floor |
|---|---|---|
| GPU | 1× RTX PRO 6000 Blackwell, 98 GB | ≥ 48 GB (LoRA bf16) or ≥ 80 GB (Full FT bf16, batch=1) |
| CUDA | 13.0 / driver 580.x | 12.1+ |
| Python | 3.12 | 3.10+ |
| Disk | ≥ 100 GB free | one Full FT ckpt ≈ 15 GB |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Verify (must match exactly):
```bash
python3 -c "import torch, transformers, peft, trl, torchao; \
print(torch.__version__, transformers.__version__, peft.__version__, trl.__version__, torchao.__version__)"
# 2.10.0+cu128 4.57.6 0.19.1 1.4.0 0.17.0
```

> **Do not use transformers 5.x.** 5.x removes `ROPE_INIT_FUNCTIONS['default']`
> and changes `DynamicCache`; Fast-dLLM v2 trust_remote_code breaks. Pin 4.57.6.

---

## 3. Compatibility patches (already applied — do not revert)

| # | Issue | Where |
|---|---|---|
| Q1 | `apply_chat_template(tokenize=True)` may return `BatchEncoding` | `_tok()` in `train/train_fastdllm.py` |
| Q2 | `gradient_checkpointing=True` + bf16 + Fast-dLLM custom forward → NaN loss | disabled in `train/train_fastdllm.py:144` |
| Q3 | `DynamicCache.key_cache[i]` removed → use `.layers[i].keys/.values` | `eval/fastdllm_generation.py:155` |
| Q4 | `batch_sample` crashes when `prompt_len <= block_size` (past_key_values None) | guard `eval/fastdllm_generation.py:153` |
| Q5 | `trl.SFTTrainer` removed `tokenizer=` → use `processing_class=` | `train/train_qwen_v4.py:74` |

---

## 4. Data

```bash
python3 prepare_dataset.py --sizes 1k,10k,100k
```
Writes `data/{train,val,test}_{1k,10k,100k}.jsonl`. Schema per line:
```json
{"instruction": "Translate to Finnish: <EN>", "output": "<FI>", "en": "<EN>", "fi": "<FI>"}
```
New language pair → edit constants at the top of `prepare_dataset.py`. Do not fork.

---

## 5. Train + eval

One run = one `outputs/<run>/` directory. One run = one `eval_*.json` per mode.

### 5.1) Reference NMT baseline (< 1 min)
```bash
python3 eval/eval_reference.py
```

### 5.2) Qwen2.5-7B + LoRA (≈3 h)
```bash
python3 train/train_qwen_v4.py \
  --train_file data/train_100k.jsonl --val_file data/val_1k.jsonl \
  --output_dir outputs/qwen_100k_lora256 \
  --target_modules all-linear --lora_rank 256 --lora_alpha 512 \
  --lr 2e-4 --epochs 1
python3 eval/eval_qwen.py --adapter outputs/qwen_100k_lora256/adapter \
  --mode free      --out outputs/qwen_100k_lora256/eval_free.json
python3 eval/eval_qwen.py --adapter outputs/qwen_100k_lora256/adapter \
  --mode gt_length --out outputs/qwen_100k_lora256/eval_gt_length.json
```
`--lora_rank 0` switches the same script to Full FT (batch=1, grad_accum=8).

### 5.3) Fast-dLLM v2 7B — Full FT (≈3 h)
```bash
python3 train/train_fastdllm.py \
  --train_file data/train_100k.jsonl \
  --output_dir outputs/fastdllm_v2_100k_fullft \
  --lr 2e-5 --epochs 1 --batch_size 1 --grad_accum 8 --max_len 512 \
  --lora_rank 0
python3 eval/eval_fastdllm.py --ckpt outputs/fastdllm_v2_100k_fullft/final \
  --mode free      --out outputs/fastdllm_v2_100k_fullft/eval_free.json
python3 eval/eval_fastdllm.py --ckpt outputs/fastdllm_v2_100k_fullft/final \
  --mode gt_length --out outputs/fastdllm_v2_100k_fullft/eval_gt_length.json
```

### 5.4) Fast-dLLM v2 7B — LoRA
```bash
python3 train/train_fastdllm.py \
  --train_file data/train_100k.jsonl \
  --output_dir outputs/fastdllm_v2_100k_lora256 \
  --lr 5e-5 --epochs 1 --batch_size 1 --grad_accum 8 --max_len 512 \
  --lora_rank 256 --lora_alpha 512 --lora_target all-linear
python3 eval/eval_fastdllm.py --ckpt outputs/fastdllm_v2_100k_lora256/final \
  --mode free      --out outputs/fastdllm_v2_100k_lora256/eval_free.json
python3 eval/eval_fastdllm.py --ckpt outputs/fastdllm_v2_100k_lora256/final \
  --mode gt_length --out outputs/fastdllm_v2_100k_lora256/eval_gt_length.json
```

---

## 6. Eval modes — exact semantics

| Mode | `max_new_tokens` | Stop / cut rule | Use |
|---|---|---|---|
| `free` | 256 (cap) | first EOS | natural generation |
| `gt_length` | `len_tok(fi_ref)` (Qwen) or block-aligned (Fast-dLLM) | truncate output to exactly `len_tok(fi_ref)`, EOS ignored | oracle-length comparison |

Report both per model. Do not mix in a single comparison row.

---

## 7. Pre-trained artifacts (Hugging Face)

| Run | Repo |
|---|---|
| Qwen2.5-7B Full FT 100k (ckpt-7500) | https://huggingface.co/sungkwang2/qwen2.5-7b-en-fi-fullft-100k |
| Fast-dLLM v2 7B Full FT 100k | https://huggingface.co/sungkwang2/fastdllm-v2-7b-en-fi-fullft-100k |
| Fast-dLLM v2 7B LoRA r=256 100k | https://huggingface.co/sungkwang2/fastdllm-v2-7b-en-fi-lora256-100k |

Eval directly from HF id (Fast-dLLM Full FT example):
```bash
python3 eval/eval_fastdllm.py --ckpt sungkwang2/fastdllm-v2-7b-en-fi-fullft-100k \
  --mode free --out outputs/<run>/eval_free.json
```

---

## 8. Roadmap

- [x] Fast-dLLM v2 Full FT 100k × 1 ep (free)
- [x] Fast-dLLM v2 LoRA r=256 100k × 1 ep (free)
- [x] Qwen Full FT 100k × 1 ep (75%, ckpt-7500, free)
- [ ] Qwen LoRA r=256 100k × 1 ep
- [ ] `gt_length` eval for all rows
- [ ] Error analysis (≥3 axes from `reports/REPORT_GUIDE.md`)

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `KeyError: 'default'` in ROPE | confirm transformers 4.57.6 |
| `'DynamicCache' object has no attribute 'key_cache'` | confirm Q3 |
| `SFTTrainer ... unexpected keyword 'tokenizer'` | confirm Q5 |
| Fast-dLLM training loss NaN from step 1 | confirm Q2 (`gradient_checkpointing=False`) |
| Eval crash inside `batch_sample` (`past_key_values is None`) | confirm Q4 |
| CUDA OOM on Fast-dLLM Full FT | `--grad_accum 16`; or switch to LoRA |
| Disk fills during training | raise `--save_steps`; default 4000 (2 ckpts over 10k steps) + `save_total_limit=2` already applied |

---

## References

- Fast-dLLM v2 7B — https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B
- Qwen2.5-7B-Instruct — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- OPUS-100 — https://huggingface.co/datasets/Helsinki-NLP/opus-100
- Reference NMT — https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-fi
