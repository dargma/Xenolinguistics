# Install & Run — single source of truth for commands

> AI-assistant guide. When a teammate asks you to install, reproduce, or run
> anything in this repo, the canonical commands live here. Do not invent
> variants. If a command fails, jump to §8 troubleshooting before mutating
> the call.

---

## 1. Environment contract

| Item | Verified | Floor |
|---|---|---|
| GPU | 1× RTX PRO 6000 Blackwell, 98 GB | 1× ≥ 48 GB (LoRA, bf16) or ≥ 80 GB (Full FT) |
| CUDA | 13.0 / driver 580.x | 12.1+ |
| Python | 3.12 | 3.10+ |
| Disk | ≥ 50 GB free (one Full-FT checkpoint = ~15 GB) | |

If the teammate's environment fails this contract, say so explicitly before
running. Do not attempt to "downgrade" silently.

## 2. Install

```bash
# 2.1) Python env
python3 -m venv .venv && source .venv/bin/activate

# 2.2) PyTorch (skip if a working torch is already present)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# 2.3) Project deps
pip install -r requirements.txt
```

Verify (run, then compare against the expected output below):
```bash
python3 - <<'PY'
import torch, transformers, peft, trl, torchao
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("peft:", peft.__version__, "trl:", trl.__version__, "torchao:", torchao.__version__)
PY
```
Expected:
```
torch: 2.10.0+cu128  cuda: True
transformers: 5.8.1
peft: 0.19.1  trl: 1.4.0  torchao: 0.17.0
```
Any mismatch → §8.

## 3. Mandatory compatibility quirks (read before running training)

These are facts of the Fast-dLLM v2 + transformers 5.x stack. Already
applied in this repo. Do **not** revert them.

| # | Issue | Where the fix lives |
|---|---|---|
| Q1 | `modeling.py` references `ROPE_INIT_FUNCTIONS['default']`, removed in transformers 5.x | Monkey-patch at top of `train_fastdllm.py` |
| Q2 | `apply_chat_template(tokenize=True)` returns `BatchEncoding`, not `list[int]`, in transformers 5.x | `_tok()` helper in `train_fastdllm.py` |
| Q3 | `peft` LoRA injection on Fast-dLLM v2 needs `torchao >= 0.16` | Pinned `>= 0.17.0` in `requirements.txt` |
| Q4 | `batch_sample` crashes when `prompt_len <= block_size` (past_key_values is None) | Guard at `fastdllm_generation.py:153` |

If a teammate's script loads Fast-dLLM v2 outside this repo, copy Q1 and Q2
patches at the top of their script. Do not edit transformers source.

## 4. Smoke test (configs only, no weight downloads)

Run this once before any training:
```bash
python3 - <<'PY'
from transformers import AutoConfig
for name, kwargs in [
    ("Qwen/Qwen2.5-7B-Instruct", {}),
    ("Efficient-Large-Model/Fast_dLLM_v2_7B", {"trust_remote_code": True}),
    ("Helsinki-NLP/opus-mt-tc-big-en-fi", {}),
]:
    cfg = AutoConfig.from_pretrained(name, **kwargs)
    print("OK", name, cfg.model_type)
PY
```
Three `OK` lines → proceed. Any error → §8.

## 5. Data

```bash
python3 prepare_dataset.py --sizes 1k,10k,100k
```
Produces `data/{train,val,test}_{1k,10k,100k}.jsonl`. Schema:
```json
{"instruction": "Translate to Finnish: <EN>", "output": "<FI>", "en": "<EN>", "fi": "<FI>"}
```

New language pair: edit `LANG_PAIR`, `SRC`, `TGT`, `LANG_NAME` at the top of
`prepare_dataset.py`. **Do not** fork the script.

## 6. Train + eval (canonical commands)

Each run writes to one `outputs/<run>/` directory. One run → one
`eval_results.json`. One run → one report under `reports/`.

### 6.1) Reference NMT baseline (no training, ~1 min)
```bash
python3 eval_reference.py
```
Writes `outputs/reference/eval_results.json`.

### 6.2) Qwen2.5-7B-Instruct + LoRA

Recommended configs (avoid the 100k × 3 ep overfit):

```bash
# A. 100k × 1 epoch  (~3 h on 1× RTX PRO 6000)
python3 train_qwen_v4.py \
  --train_file data/train_100k.jsonl \
  --val_file   data/val_100k.jsonl \
  --output_dir outputs/qwen_100k_ep1 \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --lora_rank 16 --lora_alpha 32 \
  --lr 2e-4 --epochs 1

# B. 10k × 3 epochs  (~25 min)
python3 train_qwen_v4.py \
  --train_file data/train_10k.jsonl \
  --val_file   data/val_10k.jsonl \
  --output_dir outputs/qwen_10k \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --lora_rank 16 --lora_alpha 32 \
  --lr 2e-4 --epochs 3

# Eval (greedy, max_new_tokens=128)
python3 eval_qwen.py --adapter outputs/<run>/adapter
```

Do **not** run 100k × 3 ep — produces the overfit row at
`outputs/qwen_100k/` (chrF 17.05).

### 6.3) Fast-dLLM v2 7B — Full FT
```bash
python3 train_fastdllm.py \
  --train_file data/train_10k.jsonl \
  --output_dir outputs/fastdllm_v2_10k \
  --lr 2e-5 --epochs 1 --batch_size 1 --grad_accum 8 --max_len 512

python3 eval_fastdllm.py \
  --ckpt outputs/fastdllm_v2_10k/final \
  --out  outputs/fastdllm_v2_10k/eval_results.json
# default --mode gt_length (oracle target length). Pass --mode free for the supplementary number.
```
Wall time: ~32 min on 1× RTX PRO 6000.

### 6.4) Fast-dLLM v2 7B — LoRA
```bash
python3 train_fastdllm.py \
  --train_file data/train_100k.jsonl \
  --output_dir outputs/fastdllm_v2_100k_lora64 \
  --lr 2e-4 --epochs 1 \
  --lora_rank 64 --lora_alpha 128

python3 eval_fastdllm.py \
  --ckpt outputs/fastdllm_v2_100k_lora64/final \
  --out  outputs/fastdllm_v2_100k_lora64/eval_results.json
```

## 7. Use our pre-trained artifacts from HuggingFace

After we push the artifacts:

```python
# Qwen LoRA adapter
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                            torch_dtype="bfloat16",
                                            device_map="auto")
model = PeftModel.from_pretrained(base, "<our-hf-org>/qwen-en-fi-lora-100k")
tok = AutoTokenizer.from_pretrained("<our-hf-org>/qwen-en-fi-lora-100k")
```

For Fast-dLLM v2 (Full FT weights), pass the HF id directly:
```bash
python3 eval_fastdllm.py --ckpt <our-hf-org>/fastdllm-v2-7b-en-fi-fullft-10k \
  --out outputs/<run>/eval_results.json
```

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `KeyError: 'default'` in ROPE | Confirm monkey-patch at the top of `train_fastdllm.py` is intact |
| `too many dimensions 'str'` during data collation | `apply_chat_template` returned a string. Confirm `_tok()` helper is present |
| `ImportError: torchao` version | `pip install -U "torchao>=0.17.0"` |
| `mask_token_id is None` | `tokenizer.add_special_tokens({'mask_token':'|<MASK>|'})` then `model.resize_token_embeddings(len(tokenizer))` |
| CUDA OOM on Fast-dLLM Full FT | `--grad_accum 16` (or 32); or switch to `--lora_rank 64` |
| `sacrebleu` missing | `pip install sacrebleu` |
| Fast-dLLM eval crash inside `batch_sample` (`past_key_values is None`) | Confirm `fastdllm_generation.py:153` guard |
| Eval chrF surprisingly low for Diffusion LLM | Confirm `--mode gt_length` was used (default). `free` mode under-performs |
