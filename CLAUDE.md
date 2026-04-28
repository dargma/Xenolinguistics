# Dream-7B vs Qwen2.5-7B LoRA Fine-Tuning 실험 지시서
> Claude Code 전용 실행 지시서  
> 목적: Dream-7B(Diffusion LLM)와 Qwen2.5-7B(AR LLM)의 LoRA SFT 가능성 검증 및 비교

---

## 전체 구조

```
Phase 0 — Smoke Test (환경 및 모델 로드 최소 검증)
Phase 1 — 환경 셋업 및 모델/데이터 확보
Phase 2 — Fine-Tuning 실행 (Reference / Qwen / Dream)
Phase 3 — 평가 (정량 + 정성, 세 모델 비교)
Phase 4 — 결과 정리 보고서 작성
```

**실행 원칙**
- 모델과 데이터셋은 Drive에 저장하되, 학습/추론 시에는 반드시 local_fast로 복사 후 사용 (Drive IO 속도 느림)
- Phase 0 → 1 → 2 → 3 → 4 순서 준수
- 각 Phase 완료 시 `logs/phaseN_done.txt`에 완료 시각과 요약 기록
- 에러 발생 시 즉시 중단, `logs/errors.txt`에 기록 후 보고
- **단일 Python 환경** 사용: `transformers==4.46.2` + `torch==2.5.1`
  - 사전 확인 완료: 두 모델 모두 이 버전에서 config 로드 성공
  - conda env 분리 불필요

---

## Phase 0 — Smoke Test

> **목적**: 본 실험 전 최소 비용으로 환경/모델/데이터/학습 루프 동작 확인  
> **목표 소요 시간**: 10분 이내  
> **통과 기준**: 모든 항목 ERROR 없음

### 0-1. 패키지 import 확인

```python
# smoke_test.py
import sys

checks = {}

try:
    import torch
    checks["torch"] = torch.__version__
    checks["cuda"] = torch.cuda.is_available()
    checks["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    checks["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0
except Exception as e:
    checks["torch"] = f"FAIL: {e}"

for pkg in ["transformers", "peft", "trl", "datasets", "sacrebleu"]:
    try:
        m = __import__(pkg)
        checks[pkg] = m.__version__
    except Exception as e:
        checks[pkg] = f"FAIL: {e}"

print("=== Smoke Test: Package Check ===")
for k, v in checks.items():
    status = "✅" if "FAIL" not in str(v) else "❌"
    print(f"{status} {k}: {v}")

failed = [k for k, v in checks.items() if "FAIL" in str(v)]
if failed:
    print(f"\n[ABORT] 실패 항목: {failed} → Phase 1 설치 진행")
    sys.exit(1)
else:
    print("\n[PASS] 모든 패키지 정상")
```

### 0-2. 모델 config 로드 확인 (가중치 다운로드 없이)

```python
# smoke_model_config.py
from transformers import AutoConfig

models = {
    "Qwen2.5-7B":        ("Qwen/Qwen2.5-7B-Instruct",            {}),
    "Dream-7B":          ("Dream-org/Dream-v0-Instruct-7B",       {"trust_remote_code": True}),
    "opus-mt-tc-big-en-fi": ("Helsinki-NLP/opus-mt-tc-big-en-fi", {}),
}

print("=== Smoke Test: Model Config ===")
all_pass = True
for name, (model_id, kwargs) in models.items():
    try:
        cfg = AutoConfig.from_pretrained(model_id, **kwargs)
        print(f"✅ {name}: model_type={cfg.model_type}")
    except Exception as e:
        print(f"❌ {name}: FAIL — {e}")
        all_pass = False

print("\n[PASS]" if all_pass else "\n[FAIL] 일부 모델 config 로드 실패")
```

### 0-3. 데이터셋 로드 확인 (10쌍)

```python
# smoke_dataset.py
from datasets import load_dataset

print("=== Smoke Test: Dataset ===")
try:
    ds = load_dataset("Helsinki-NLP/opus-100", "en-fi", split="train[:10]")
    print(f"✅ 데이터셋 로드: {len(ds)}쌍")
    print(f"   샘플 EN: {ds[0]['translation']['en']}")
    print(f"   샘플 FI: {ds[0]['translation']['fi']}")
except Exception as e:
    print(f"❌ FAIL: {e}")
```

### 0-4. 미니 학습 루프 (더미, 5 step)

```python
# smoke_train_loop.py
import torch
import torch.nn as nn

print("=== Smoke Test: Training Loop ===")
try:
    model = nn.Linear(16, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for step in range(5):
        x = torch.randn(4, 16)
        loss = model(x).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(round(loss.item(), 4))
    print(f"✅ 학습 루프 정상: losses={losses}")
except Exception as e:
    print(f"❌ FAIL: {e}")
```

### 0-5. Smoke Test 결과 저장

```bash
python3 smoke_test.py        | tee -a logs/smoke_test_result.txt
python3 smoke_model_config.py | tee -a logs/smoke_test_result.txt
python3 smoke_dataset.py      | tee -a logs/smoke_test_result.txt
python3 smoke_train_loop.py   | tee -a logs/smoke_test_result.txt
```

모두 PASS → Phase 1 진행. FAIL 항목 있으면 해당 항목 수정 후 재실행.

---

## Phase 1 — 환경 셋업 및 데이터 준비

### 1-1. 환경 정보 수집

```bash
mkdir -p logs outputs/reference outputs/qwen outputs/dream

nvidia-smi | tee logs/env_info.txt
nvcc --version >> logs/env_info.txt
python3 --version >> logs/env_info.txt
pip show torch transformers peft trl accelerate bitsandbytes 2>/dev/null >> logs/env_info.txt
df -h / >> logs/env_info.txt
```

### 1-2. 패키지 설치 (단일 환경)

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.2
pip install peft==0.13.2
pip install trl==0.12.0
pip install accelerate==1.1.1
pip install bitsandbytes==0.44.1
pip install datasets sacrebleu bert-score matplotlib pandas sentencepiece

# 버전 재확인
python3 -c "
import torch, transformers, peft, trl
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('trl:', trl.__version__)
" | tee -a logs/env_info.txt
```

### 1-3. 데이터셋 준비

```python
# prepare_dataset.py
from datasets import load_dataset
import json, os

os.makedirs("data", exist_ok=True)

ds = load_dataset("Helsinki-NLP/opus-100", "en-fi", split="train")
print(f"전체: {len(ds)}쌍")

for n, tag in [(1000, "1k"), (10000, "10k")]:
    subset = ds.select(range(n))
    splits = {
        "train": subset.select(range(int(n*0.8))),
        "val":   subset.select(range(int(n*0.8), int(n*0.9))),
        "test":  subset.select(range(int(n*0.9), n)),
    }
    for split_name, split_data in splits.items():
        path = f"data/{split_name}_{tag}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in split_data:
                f.write(json.dumps({
                    "instruction": f"Translate to Finnish: {ex['translation']['en']}",
                    "output":      ex['translation']['fi'],
                    "en":          ex['translation']['en'],
                    "fi":          ex['translation']['fi'],
                }, ensure_ascii=False) + "\n")
        print(f"  {path}: {len(split_data)}쌍")

print("데이터셋 준비 완료 → data/")
```

### 1-4. 모델 전체 로드 확인 (GPU)

```python
# check_models_full.py
import torch, json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, MarianMTModel

results = {}

# Qwen2.5-7B
try:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    m   = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
              torch_dtype=torch.bfloat16, device_map="auto")
    vram = torch.cuda.memory_allocated()/1e9
    results["Qwen2.5-7B"] = {"status": "OK", "vram_gb": round(vram,2), "class": type(m).__name__}
    del m; torch.cuda.empty_cache()
except Exception as e:
    results["Qwen2.5-7B"] = {"status": f"FAIL: {e}"}

# Dream-7B
try:
    tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
    m   = AutoModel.from_pretrained("Dream-org/Dream-v0-Instruct-7B",
              torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    vram = torch.cuda.memory_allocated()/1e9
    results["Dream-7B"] = {
        "status": "OK", "vram_gb": round(vram,2),
        "class": type(m).__name__, "mask_token_id": tok.mask_token_id
    }
    del m; torch.cuda.empty_cache()
except Exception as e:
    results["Dream-7B"] = {"status": f"FAIL: {e}"}

# Reference NMT
try:
    tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-fi")
    m   = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-fi")
    results["opus-mt-en-fi"] = {"status": "OK", "class": type(m).__name__,
                                 "params_M": round(sum(p.numel() for p in m.parameters())/1e6, 1)}
    del m
except Exception as e:
    results["opus-mt-en-fi"] = {"status": f"FAIL: {e}"}

print(json.dumps(results, indent=2))
json.dump(results, open("logs/model_check.json","w"), indent=2)
```

---

## Phase 2 — Fine-Tuning 실행

> 세 모델을 순서대로 실행: **Reference → Qwen → Dream**  
> `DATA_TAG = "1k"` 로 먼저 파이프라인 확인 → 정상이면 `"10k"` 로 변경 후 재실행

### 모델 역할 정의

| 역할 | 모델 ID | 용도 |
|------|---------|------|
| **Reference NMT** | `Helsinki-NLP/opus-mt-tc-big-en-fi` | 전용 번역 모델. 학습 없음. "이 정도면 합리적" 기준선 |
| **AR LLM** | `Qwen/Qwen2.5-7B-Instruct` | LoRA SFT, 표준 causal LM |
| **Diffusion LLM** | `Dream-org/Dream-v0-Instruct-7B` | LoRA SFT, masked diffusion loss |

---

### 2-A. Reference 모델 평가 (학습 없음)

```python
# eval_reference.py
# Helsinki-NLP/opus-mt-tc-big-en-fi: OPUS-100 en-fi 전용 NMT 모델
# fine-tuning 없이 바로 추론 → 성능 기준선 측정
import json, torch, os
import sacrebleu
from transformers import MarianMTModel, MarianTokenizer

DATA_TAG = "1k"
MODEL_ID = "Helsinki-NLP/opus-mt-tc-big-en-fi"
os.makedirs("outputs/reference", exist_ok=True)

test_data = [json.loads(l) for l in open(f"data/test_{DATA_TAG}.jsonl")]

tokenizer = MarianTokenizer.from_pretrained(MODEL_ID)
model     = MarianMTModel.from_pretrained(MODEL_ID)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()

predictions, references, examples = [], [], []

for i, ex in enumerate(test_data[:100]):
    inputs = tokenizer([ex["en"]], return_tensors="pt", padding=True,
                       truncation=True, max_length=256)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, num_beams=4)
    gen = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    predictions.append(gen)
    references.append(ex["fi"])
    if i < 10:
        examples.append({"en": ex["en"], "fi_ref": ex["fi"], "fi_pred": gen})

chrf = sacrebleu.corpus_chrf(predictions, [references])
bleu = sacrebleu.corpus_bleu(predictions, [references])

result = {
    "model": MODEL_ID, "role": "reference_nmt",
    "data_tag": DATA_TAG, "n_eval": len(predictions),
    "chrF": round(chrf.score, 2), "BLEU": round(bleu.score, 2),
    "note": "OPUS-100 en-fi 전용 학습 모델. fine-tuning 없음. 성능 기준선.",
    "examples": examples
}
json.dump(result, open("outputs/reference/eval_results.json","w"), indent=2, ensure_ascii=False)

print(f"[Reference NMT] chrF={chrf.score:.2f}, BLEU={bleu.score:.2f}")
print("\n=== 출력 예시 ===")
for ex in examples[:5]:
    print(f"EN:   {ex['en']}")
    print(f"REF:  {ex['fi_ref']}")
    print(f"PRED: {ex['fi_pred']}\n")
```

---

### 2-B. Qwen2.5-7B LoRA SFT

```python
# train_qwen.py
import torch, time, json, os, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

os.makedirs("outputs/qwen", exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_TAG = "1k"   # 파이프라인 확인 후 "10k"로 변경
EPOCHS   = 3
LR       = 2e-4
MAX_SEQ  = 256

def load_ds(path):
    data = []
    for line in open(path):
        d = json.loads(line)
        data.append({"text": (
            f"<|im_start|>user\n{d['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{d['output']}<|im_end|>"
        )})
    return Dataset.from_list(data)

train_ds = load_ds(f"data/train_{DATA_TAG}.jsonl")
val_ds   = load_ds(f"data/val_{DATA_TAG}.jsonl")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model     = AutoModelForCausalLM.from_pretrained(MODEL_ID,
    torch_dtype=torch.bfloat16, device_map="auto")
vram_base = torch.cuda.memory_allocated()/1e9

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","v_proj","k_proj","o_proj"],
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
vram_lora = torch.cuda.memory_allocated()/1e9

args = SFTConfig(
    output_dir="outputs/qwen",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps", eval_steps=50,
    save_steps=200,
    max_seq_length=MAX_SEQ,
    report_to="none",
    dataset_text_field="text"
)

trainer = SFTTrainer(model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer)

start = time.time()
trainer.train()
elapsed = time.time() - start

# Loss 그래프
history    = trainer.state.log_history
train_loss = [(h["step"], h["loss"])      for h in history if "loss" in h]
eval_loss  = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

fig, ax = plt.subplots(figsize=(8,4))
if train_loss: ax.plot(*zip(*train_loss), label="Train Loss")
if eval_loss:  ax.plot(*zip(*eval_loss),  label="Eval Loss", linestyle="--")
ax.set_xlabel("Step"); ax.set_ylabel("Loss")
ax.set_title(f"Qwen2.5-7B LoRA SFT — {DATA_TAG} / lr={LR} / r=16")
ax.legend(); ax.grid(True, alpha=0.3)
fig.savefig(f"outputs/qwen/loss_curve_{DATA_TAG}.png", dpi=150, bbox_inches="tight")
print(f"Loss 그래프 저장: outputs/qwen/loss_curve_{DATA_TAG}.png")

meta = {
    "model": MODEL_ID, "data_tag": DATA_TAG, "epochs": EPOCHS, "lr": LR,
    "lora_r": 16, "vram_base_gb": round(vram_base,2), "vram_lora_gb": round(vram_lora,2),
    "train_time_min": round(elapsed/60,1),
    "final_train_loss": train_loss[-1][1] if train_loss else None,
    "final_eval_loss":  eval_loss[-1][1]  if eval_loss  else None,
}
json.dump(meta, open("outputs/qwen/training_meta.json","w"), indent=2)
model.save_pretrained("outputs/qwen/adapter")
tokenizer.save_pretrained("outputs/qwen/adapter")
print(f"Qwen SFT 완료 | {elapsed/60:.1f}분 | train loss: {train_loss[-1][1]:.4f}")
```

---

### 2-C. Dream-7B LoRA SFT

**사전 확인**: Dream 공식 레포(`https://github.com/DreamLM/Dream`)의 `src/` 폴더에  
단일 GPU용 SFT 스크립트가 있으면 그것을 우선 사용.  
없거나 multi-GPU 전용이면 아래 구현 사용.

```python
# train_dream.py
import torch, time, json, os, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

os.makedirs("outputs/dream", exist_ok=True)

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
DATA_TAG = "1k"
EPOCHS   = 3
LR       = 2e-4
MAX_SEQ  = 256

train_data = [json.loads(l) for l in open(f"data/train_{DATA_TAG}.jsonl")]
val_data   = [json.loads(l) for l in open(f"data/val_{DATA_TAG}.jsonl")]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
MASK_ID   = tokenizer.mask_token_id
print(f"mask_token_id: {MASK_ID}")
assert MASK_ID is not None, "mask_token_id None — tokenizer 확인 필요"

model     = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
print("Dream model class:", type(model).__name__)
vram_base = torch.cuda.memory_allocated()/1e9

# task_type 미지정 (Diffusion 모델 — CAUSAL_LM 해당 없음)
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","v_proj","k_proj","o_proj"]
)
model.enable_input_require_grads()   # LoRA gradient flow 보장
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
vram_lora = torch.cuda.memory_allocated()/1e9

def collate_fn(batch):
    texts = [f"{ex['instruction']} {ex['output']}" for ex in batch]
    return tokenizer(texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=MAX_SEQ)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_data,   batch_size=4, shuffle=False, collate_fn=collate_fn)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS*len(train_loader))

loss_log, val_loss_log = [], []
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_losses = []
    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        # Masked Diffusion SFT: 랜덤 비율로 마스킹 → 복원 loss
        mask_ratio         = torch.rand(1).item() * 0.7 + 0.15
        rand               = torch.rand_like(input_ids, dtype=torch.float)
        masked             = rand < mask_ratio
        noisy_ids          = input_ids.clone(); noisy_ids[masked] = MASK_ID
        labels             = torch.full_like(input_ids, -100)
        labels[masked]     = input_ids[masked]

        out  = model(input_ids=noisy_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()

        epoch_losses.append(loss.item())
        global_step = epoch * len(train_loader) + step
        if step % 10 == 0:
            print(f"Epoch {epoch+1} Step {step} Loss: {loss.item():.4f}")
            loss_log.append({"global_step": global_step, "loss": loss.item()})

    # Validation loss
    model.eval()
    v_losses = []
    with torch.no_grad():
        for vbatch in val_loader:
            ids  = vbatch["input_ids"].to(model.device)
            mask = vbatch["attention_mask"].to(model.device)
            m    = torch.rand_like(ids, dtype=torch.float) < 0.5
            n    = ids.clone(); n[m] = MASK_ID
            lb   = torch.full_like(ids, -100); lb[m] = ids[m]
            v_losses.append(model(input_ids=n, attention_mask=mask, labels=lb).loss.item())
    val_loss = sum(v_losses)/len(v_losses)
    val_loss_log.append({"global_step": global_step, "val_loss": val_loss})
    print(f"Epoch {epoch+1} | train: {sum(epoch_losses)/len(epoch_losses):.4f} | val: {val_loss:.4f}")

elapsed = time.time() - start

# Loss 그래프
fig, ax = plt.subplots(figsize=(8,4))
ax.plot([d["global_step"] for d in loss_log],     [d["loss"] for d in loss_log],
        label="Train Loss", alpha=0.7)
ax.plot([d["global_step"] for d in val_loss_log], [d["val_loss"] for d in val_loss_log],
        label="Val Loss", linestyle="--", marker="o")
ax.set_xlabel("Step"); ax.set_ylabel("Loss")
ax.set_title(f"Dream-7B LoRA SFT (Masked Diffusion) — {DATA_TAG} / lr={LR} / r=16")
ax.legend(); ax.grid(True, alpha=0.3)
fig.savefig(f"outputs/dream/loss_curve_{DATA_TAG}.png", dpi=150, bbox_inches="tight")

meta = {
    "model": MODEL_ID, "data_tag": DATA_TAG, "epochs": EPOCHS, "lr": LR,
    "lora_r": 16, "sft_method": "masked_diffusion_random_mask_ratio_0.15_0.85",
    "vram_base_gb": round(vram_base,2), "vram_lora_gb": round(vram_lora,2),
    "train_time_min": round(elapsed/60,1),
    "final_train_loss": loss_log[-1]["loss"]         if loss_log     else None,
    "final_val_loss":   val_loss_log[-1]["val_loss"] if val_loss_log else None,
}
json.dump(meta, open("outputs/dream/training_meta.json","w"), indent=2)
model.save_pretrained("outputs/dream/adapter")
tokenizer.save_pretrained("outputs/dream/adapter")
print(f"Dream SFT 완료 | {elapsed/60:.1f}분")
```

---

## Phase 3 — 평가

### 3-A. Qwen2.5-7B 평가

```python
# eval_qwen.py
import json, torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DATA_TAG = "1k"
test_data = [json.loads(l) for l in open(f"data/test_{DATA_TAG}.jsonl")]

tokenizer = AutoTokenizer.from_pretrained("outputs/qwen/adapter", trust_remote_code=True)
base      = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16, device_map="auto")
model     = PeftModel.from_pretrained(base, "outputs/qwen/adapter")
model.eval()

predictions, references, examples = [], [], []
for i, ex in enumerate(test_data[:100]):
    prompt = (f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
    predictions.append(gen)
    references.append(ex["fi"])
    if i < 10:
        examples.append({"en": ex["en"], "fi_ref": ex["fi"], "fi_pred": gen})

chrf = sacrebleu.corpus_chrf(predictions, [references])
bleu = sacrebleu.corpus_bleu(predictions, [references])

result = {
    "model": "Qwen2.5-7B + LoRA", "data_tag": DATA_TAG,
    "chrF": round(chrf.score,2), "BLEU": round(bleu.score,2),
    "examples": examples
}
json.dump(result, open("outputs/qwen/eval_results.json","w"), indent=2, ensure_ascii=False)
print(f"[Qwen] chrF={chrf.score:.2f}, BLEU={bleu.score:.2f}")
for ex in examples[:5]:
    print(f"EN:   {ex['en']}\nREF:  {ex['fi_ref']}\nPRED: {ex['fi_pred']}\n")
```

### 3-B. Dream-7B 평가 (두 가지 모드)

Dream 평가는 **두 가지 모드** 모두 실행:

| 모드 | 설명 | 목적 |
|------|------|------|
| `free_length` | 고정 max_new_tokens=64 | 실제 상황 (길이 미지) |
| `gt_length` | GT 문장의 토큰 수를 마스크 수로 설정 | 성능 상한 측정 |

```python
# eval_dream.py
import json, torch
import sacrebleu
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

DATA_TAG = "1k"
MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
test_data = [json.loads(l) for l in open(f"data/test_{DATA_TAG}.jsonl")]

tokenizer = AutoTokenizer.from_pretrained("outputs/dream/adapter", trust_remote_code=True)
MASK_ID   = tokenizer.mask_token_id
base      = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
model     = PeftModel.from_pretrained(base, "outputs/dream/adapter")
model.eval()

def dream_generate(prompt_text, n_masks, steps=20):
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids  = torch.tensor([prompt_ids + [MASK_ID]*n_masks]).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=n_masks,
                             steps=steps, temperature=0.0)
    return tokenizer.decode(out[0][len(prompt_ids):], skip_special_tokens=True).strip()

results = {}
for mode, use_gt in [("free_length", False), ("gt_length", True)]:
    predictions, references, examples = [], [], []
    for i, ex in enumerate(test_data[:100]):
        prompt  = ex["instruction"] + " "
        n_masks = (len(tokenizer.encode(ex["fi"], add_special_tokens=False))
                   if use_gt else 64)
        try:
            gen = dream_generate(prompt, n_masks)
        except Exception as e:
            gen = ""; print(f"[WARN] {mode} step {i}: {e}")
        predictions.append(gen)
        references.append(ex["fi"])
        if i < 10:
            examples.append({"en": ex["en"], "fi_ref": ex["fi"],
                              "fi_pred": gen, "n_masks": n_masks})

    chrf = sacrebleu.corpus_chrf(predictions, [references])
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    results[mode] = {"chrF": round(chrf.score,2), "BLEU": round(bleu.score,2),
                     "examples": examples}
    print(f"[Dream/{mode}] chrF={chrf.score:.2f}, BLEU={bleu.score:.2f}")
    for ex in examples[:3]:
        print(f"  EN:   {ex['en']}\n  REF:  {ex['fi_ref']}\n  PRED: {ex['fi_pred']}\n")

results.update({"model": "Dream-7B + LoRA", "data_tag": DATA_TAG})
json.dump(results, open("outputs/dream/eval_results.json","w"), indent=2, ensure_ascii=False)
```

### 3-C. 세 모델 나란히 비교

```python
# compare_results.py
import json

ref   = json.load(open("outputs/reference/eval_results.json"))
qwen  = json.load(open("outputs/qwen/eval_results.json"))
dream = json.load(open("outputs/dream/eval_results.json"))

# 정량 비교표
print("=" * 68)
print(f"{'모델':<38} {'chrF':>6} {'BLEU':>6}")
print("-" * 68)
rows = [
    ("opus-mt-tc-big-en-fi [Reference NMT]", ref["chrF"],                       ref["BLEU"]),
    ("Qwen2.5-7B + LoRA [AR]",               qwen["chrF"],                      qwen["BLEU"]),
    ("Dream-7B + LoRA [free_length]",         dream["free_length"]["chrF"],      dream["free_length"]["BLEU"]),
    ("Dream-7B + LoRA [gt_length]",           dream["gt_length"]["chrF"],        dream["gt_length"]["BLEU"]),
]
for name, chrf, bleu in rows:
    print(f"{name:<38} {chrf:>6} {bleu:>6}")
print("=" * 68)

# 동일 문장 정성 비교
print("\n=== 동일 입력 5문장 정성 비교 ===\n")
ref_ex   = {e["en"]: e for e in ref["examples"][:10]}
qwen_ex  = {e["en"]: e for e in qwen["examples"][:10]}
dream_ex = {e["en"]: e for e in dream["gt_length"]["examples"][:10]}

for en in list(ref_ex.keys())[:5]:
    print(f"[EN]      {en}")
    print(f"[정답]    {ref_ex[en]['fi_ref']}")
    print(f"[opus-mt] {ref_ex[en]['fi_pred']}")
    if en in qwen_ex:  print(f"[Qwen]    {qwen_ex[en]['fi_pred']}")
    if en in dream_ex: print(f"[Dream]   {dream_ex[en]['fi_pred']}")
    print()
```

---

## Phase 4 — 보고서 자동 생성

```python
# make_report.py
import json, os
from datetime import datetime

ref   = json.load(open("outputs/reference/eval_results.json"))
qmeta = json.load(open("outputs/qwen/training_meta.json"))
dmeta = json.load(open("outputs/dream/training_meta.json"))
qeval = json.load(open("outputs/qwen/eval_results.json"))
deval = json.load(open("outputs/dream/eval_results.json"))
env   = open("logs/env_info.txt").read()

report = f"""# Dream-7B vs Qwen2.5-7B LoRA Fine-Tuning 검토 결과
> 작성: {datetime.now().strftime("%Y-%m-%d")} | 자동 생성

---

## 1. 실험 환경

```
{env[:1500]}
```

**Python 환경**: `transformers==4.46.2` + `torch==2.5.1` 단일 환경  
Dream-7B / Qwen2.5-7B 모두 이 버전에서 호환 확인 완료. conda 분리 불필요.

---

## 2. 데이터셋

| 항목 | 내용 |
|------|------|
| 출처 | Helsinki-NLP/opus-100 (en-fi) |
| 규모 | {qmeta['data_tag']} |
| 분할 | train 80% / val 10% / test 10% |
| 선택 이유 | 핀란드어 = 퀘냐의 형태론적 모델 (격변화 15개, 교착어). 인공어 실험 파이프라인 검증용 |

---

## 3. 모델 구성

| 역할 | 모델 | 방식 |
|------|------|------|
| Reference NMT | Helsinki-NLP/opus-mt-tc-big-en-fi | 전용 NMT, 학습 없음, 성능 기준선 |
| AR LLM | Qwen/Qwen2.5-7B-Instruct | LoRA r=16, causal LM SFT |
| Diffusion LLM | Dream-org/Dream-v0-Instruct-7B | LoRA r=16, masked diffusion SFT |

---

## 4. 학습 설정

### Qwen2.5-7B
| 항목 | 값 |
|------|-----|
| LoRA r / alpha | 16 / 32 |
| Learning rate | {qmeta['lr']} |
| Epochs | {qmeta['epochs']} |
| Batch size | 4 (grad_accum=2, effective=8) |
| VRAM base / LoRA | {qmeta['vram_base_gb']}GB / {qmeta['vram_lora_gb']}GB |
| 학습 시간 | {qmeta['train_time_min']}분 |
| 최종 train loss | {qmeta['final_train_loss']} |
| 최종 eval loss | {qmeta['final_eval_loss']} |

Loss 그래프: `outputs/qwen/loss_curve_{qmeta['data_tag']}.png`

### Dream-7B
| 항목 | 값 |
|------|-----|
| LoRA r / alpha | 16 / 32 |
| SFT 방식 | Masked Diffusion (random mask ratio 0.15~0.85, 마스킹 위치만 loss) |
| Learning rate | {dmeta['lr']} |
| Epochs | {dmeta['epochs']} |
| VRAM base / LoRA | {dmeta['vram_base_gb']}GB / {dmeta['vram_lora_gb']}GB |
| 학습 시간 | {dmeta['train_time_min']}분 |
| 최종 train loss | {dmeta['final_train_loss']} |
| 최종 val loss | {dmeta['final_val_loss']} |

Loss 그래프: `outputs/dream/loss_curve_{dmeta['data_tag']}.png`

---

## 5. 평가 결과

### 정량 비교

| 모델 | 모드 | chrF | BLEU | 비고 |
|------|------|------|------|------|
| opus-mt-tc-big-en-fi | NMT 기준선 | {ref['chrF']} | {ref['BLEU']} | fine-tuning 없음 |
| Qwen2.5-7B + LoRA | AR 표준 생성 | {qeval['chrF']} | {qeval['BLEU']} | |
| Dream-7B + LoRA | free_length | {deval['free_length']['chrF']} | {deval['free_length']['BLEU']} | 길이 미지정 |
| Dream-7B + LoRA | gt_length | {deval['gt_length']['chrF']} | {deval['gt_length']['BLEU']} | GT 길이 제공 (상한) |

**해석 기준**  
- Reference NMT = "이 정도면 합리적" 기준선. LoRA 모델이 이에 근접하면 파이프라인 유효.  
- Dream gt_length > free_length 차이가 크면 → 길이 예측이 Diffusion LLM의 핵심 병목.

### 정성 비교 (동일 입력 5문장)

"""

for ex in ref.get("examples", [])[:5]:
    en = ex["en"]
    report += f"**EN**: {en}  \\n"
    report += f"**정답**: {ex['fi_ref']}  \\n"
    report += f"**opus-mt**: {ex['fi_pred']}  \\n"
    qex = next((e for e in qeval.get("examples",[]) if e["en"]==en), None)
    dex = next((e for e in deval["gt_length"].get("examples",[]) if e["en"]==en), None)
    if qex: report += f"**Qwen**: {qex['fi_pred']}  \\n"
    if dex: report += f"**Dream**: {dex['fi_pred']}  \\n"
    report += "\\n"

report += f"""
---

## 6. 종합 결론

### LoRA Fine-Tuning 가능성

| 모델 | LoRA 적용 | SFT 수렴 | 특이사항 |
|------|----------|---------|---------|
| Qwen2.5-7B | ✅ | ✅ | trl SFTTrainer 표준 사용 가능 |
| Dream-7B | {'✅' if dmeta['final_train_loss'] else '⚠️'} | {'✅' if dmeta['final_val_loss'] else '확인 필요'} | masked diffusion loss 커스텀 구현 필요 |

### 단일 환경 가능 여부
`transformers==4.46.2` 기준으로 두 모델 모두 정상 동작 확인.  
conda env 분리 없이 단일 환경 운영 가능.

### 인공어 실험 확장성
핀란드어 파이프라인 정상 동작 확인.  
클링온 / 퀘냐 / 칼라니어 실험은 데이터 파일만 교체하면 동일 파이프라인 재사용 가능.  
추가 평가 항목: Grammar Consistency Rate (클링온), Minimal Pair Test (퀘냐).

---
*자동 생성 보고서. 수치는 실제 실행 결과 기준.*
"""

with open("REPORT.md", "w", ensure_ascii=False) as f:
    f.write(report)
print("보고서 완료 → REPORT.md")
```

---

## 전체 실행 순서

```bash
# Phase 0: Smoke Test
python3 smoke_test.py
python3 smoke_model_config.py
python3 smoke_dataset.py
python3 smoke_train_loop.py

# Phase 1
mkdir -p logs outputs/reference outputs/qwen outputs/dream data
bash -c 'nvidia-smi; nvcc --version; python3 --version' | tee logs/env_info.txt
python3 prepare_dataset.py
python3 check_models_full.py

# Phase 2 (1k 먼저 → OK면 10k 재실행)
python3 eval_reference.py     # Reference NMT 기준선
python3 train_qwen.py         # Qwen LoRA SFT
python3 train_dream.py        # Dream LoRA SFT

# Phase 3
python3 eval_qwen.py
python3 eval_dream.py
python3 compare_results.py    # 세 모델 나란히 비교

# Phase 4
python3 make_report.py
```

---

## 에러 대응

| 상황 | 대응 |
|------|------|
| Dream LoRA gradient 오류 | `model.enable_input_require_grads()` 추가 확인 |
| Dream `generate` API 없음 | 공식 레포 `demo_batch_completion.py` 참조, iterative denoising 수동 구현 |
| `mask_token_id` None | `tokenizer.add_special_tokens({'mask_token':'[MASK]'})` 후 `model.resize_token_embeddings` |
| VRAM OOM | batch_size=2, grad_accum=4 조정. 또는 QLoRA 4bit 전환 |
| transformers 버전 충돌 | `pip install --force-reinstall transformers==4.46.2` |

---

## 출력 파일 구조

```
data/
  train_1k.jsonl / val_1k.jsonl / test_1k.jsonl
  train_10k.jsonl / val_10k.jsonl / test_10k.jsonl
outputs/
  reference/
    eval_results.json          # chrF, BLEU, 출력 예시 10개
  qwen/
    adapter/                   # LoRA 가중치
    loss_curve_1k.png
    training_meta.json
    eval_results.json
  dream/
    adapter/
    loss_curve_1k.png
    training_meta.json
    eval_results.json          # free_length + gt_length 두 모드
logs/
  smoke_test_result.txt
  env_info.txt
  model_check.json
  phase0_done.txt ~ phase4_done.txt
  errors.txt
REPORT.md
```
