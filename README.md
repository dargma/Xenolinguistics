# Xenolinguistics — AR LLM vs Diffusion LLM

동일한 백본(Qwen2.5-7B 계열) 위에서 **AR LLM** (`Qwen/Qwen2.5-7B-Instruct`)과
**Diffusion LLM** (`Efficient-Large-Model/Fast_dLLM_v2_7B`)을 같은 데이터로
공정 비교합니다. 첫 사례 연구는 영어→핀란드어(OPUS-100)이고, `dataset/prepare_dataset.py`
의 상수만 바꾸면 다른 언어쌍(향후 인공어 포함)으로 확장됩니다.

| 파일 | 용도 |
|---|---|
| `README.md` (이 파일) | 설치 · 명령어 · 결과 — 단일 진실원 |
| [`CLAUDE.md`](CLAUDE.md) | AI 어시스턴트 규칙 |
| [`reports/REPORT_GUIDE.md`](reports/REPORT_GUIDE.md) | 런 보고서 작성 가이드 |

---

## 1. 결과 (2026-05-17, en→fi, OPUS-100 100k)

테스트 셋 = `dataset/data/test_1k.jsonl` 첫 100문장. 메트릭 = `sacrebleu` 기본값.
각 런 결과는 `outputs/<run>/eval_free.json` (그리고 측정 시 `eval_gt_length.json`)에 저장됩니다.

| 모델 | 방식 | 학습 진행 | chrF (free) | BLEU (free) | chrF (gt_length) | BLEU (gt_length) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `opus-mt-tc-big-en-fi` | NMT 베이스라인 | — | **56.91** | **37.45** | — | — |
| Qwen2.5-7B | Full FT, lr=2e-5 | 1 ep, 7500/10000 (75%, 디스크 풀로 중단) | **47.72** | **27.65** | 미측정 | 미측정 |
| Fast-dLLM v2 7B | Full FT, lr=2e-5 | 1 ep, 10000/10000 (100%) | 40.86 | 17.22 | 미측정 | 미측정 |
| Fast-dLLM v2 7B | LoRA r=256, lr=5e-5 | 1 ep, 10000/10000 (100%) | 37.22 | 15.38 | 미측정 | 미측정 |

- `free`: 첫 EOS에서 종료, 토큰 cap 256
- `gt_length`: 정답 토큰 길이(`len_tok(fi_ref)`)로 정확히 컷, EOS 무시 (§6 참조)

같은 백본(Qwen2.5-7B 계열) → 품질 차이는 사전학습 데이터가 아니라 **생성 패러다임
(AR vs block masked diffusion)** 때문입니다.

**현 상태**: 모든 행에 대해 `free` 측정 완료. `gt_length` 통일 규칙 재측정 보류.
Qwen LoRA r=256 100k는 디스크 이슈로 두 번 실패하여 보류.

---

## 2. 환경

| 항목 | 검증값 | 최저 요구 |
|---|---|---|
| GPU | 1× RTX PRO 6000 Blackwell, 98 GB | ≥ 48 GB (LoRA bf16) / ≥ 80 GB (Full FT bf16, batch=1) |
| CUDA | 13.0 / 드라이버 580.x | 12.1+ |
| Python | 3.12 | 3.10+ |
| 디스크 | ≥ 100 GB 여유 | Full FT 체크포인트 1개 ≈ 15 GB |

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

설치 검증 (정확히 일치해야 함):
```bash
python3 -c "import torch, transformers, peft, trl, torchao; \
print(torch.__version__, transformers.__version__, peft.__version__, trl.__version__, torchao.__version__)"
# 2.10.0+cu128 4.57.6 0.19.1 1.4.0 0.17.0
```

> **transformers 5.x 사용 금지.** `ROPE_INIT_FUNCTIONS['default']` 키가 제거되고
> `DynamicCache` API가 바뀌어 Fast-dLLM v2의 `trust_remote_code` 모델이 깨집니다.
> 반드시 4.57.6에 고정.

---

## 3. 호환성 패치 (적용됨 — 되돌리지 말 것)

| # | 이슈 | 위치 |
|---|---|---|
| Q1 | `apply_chat_template(tokenize=True)`가 `BatchEncoding` 반환 가능 | `train/train_fastdllm.py`의 `_tok()` |
| Q2 | `gradient_checkpointing=True` + bf16 + Fast-dLLM 커스텀 forward → NaN loss | `train/train_fastdllm.py:144` 에서 비활성화 |
| Q3 | `DynamicCache.key_cache[i]` 제거 → `.layers[i].keys/.values` 사용 | `eval/fastdllm_generation.py:155` |
| Q4 | `batch_sample`가 `prompt_len <= block_size`일 때 past_key_values None으로 크래시 | `eval/fastdllm_generation.py:153` 가드 |
| Q5 | `trl.SFTTrainer`가 `tokenizer=` kwarg 제거 → `processing_class=` 사용 | `train/train_qwen_v4.py:74` |

---

## 4. 데이터 준비

```bash
python3 dataset/prepare_dataset.py --sizes 1k,10k,100k
```
`dataset/data/{train,val,test}_{1k,10k,100k}.jsonl` 를 생성. 각 줄 스키마:
```json
{"instruction": "Translate to Finnish: <EN>", "output": "<FI>", "en": "<EN>", "fi": "<FI>"}
```
다른 언어쌍 → `dataset/prepare_dataset.py` 상단 상수만 수정. 스크립트 포크 금지.

---

## 5. 학습 + 평가

런 하나 = `outputs/<run>/` 디렉터리 하나 = `eval_*.json` (모드별) 하나.

### 5.1) Reference NMT 베이스라인 (학습 없음, < 1분)
```bash
python3 eval/eval_reference.py
```

### 5.2) Qwen2.5-7B + LoRA (≈ 3시간)
```bash
python3 train/train_qwen_v4.py \
  --train_file dataset/data/train_100k.jsonl --val_file dataset/data/val_1k.jsonl \
  --output_dir outputs/qwen_100k_lora256 \
  --target_modules all-linear --lora_rank 256 --lora_alpha 512 \
  --lr 2e-4 --epochs 1
python3 eval/eval_qwen.py --adapter outputs/qwen_100k_lora256/adapter \
  --mode free      --out outputs/qwen_100k_lora256/eval_free.json
python3 eval/eval_qwen.py --adapter outputs/qwen_100k_lora256/adapter \
  --mode gt_length --out outputs/qwen_100k_lora256/eval_gt_length.json
```
`--lora_rank 0` 으로 같은 스크립트가 Full FT 모드 (batch=1, grad_accum=8).

### 5.3) Fast-dLLM v2 7B — Full FT (≈ 3시간)
```bash
python3 train/train_fastdllm.py \
  --train_file dataset/data/train_100k.jsonl \
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
  --train_file dataset/data/train_100k.jsonl \
  --output_dir outputs/fastdllm_v2_100k_lora256 \
  --lr 5e-5 --epochs 1 --batch_size 1 --grad_accum 8 --max_len 512 \
  --lora_rank 256 --lora_alpha 512 --lora_target all-linear
python3 eval/eval_fastdllm.py --ckpt outputs/fastdllm_v2_100k_lora256/final \
  --mode free      --out outputs/fastdllm_v2_100k_lora256/eval_free.json
python3 eval/eval_fastdllm.py --ckpt outputs/fastdllm_v2_100k_lora256/final \
  --mode gt_length --out outputs/fastdllm_v2_100k_lora256/eval_gt_length.json
```

---

## 6. 평가 모드 — 정확한 의미

| 모드 | `max_new_tokens` | 종료/컷 규칙 | 용도 |
|---|---|---|---|
| `free` | 256 (cap) | 첫 EOS | 자연 생성 |
| `gt_length` | `len_tok(fi_ref)` (Qwen) / 블록 정렬 (Fast-dLLM) | 정확히 `len_tok(fi_ref)` 토큰만큼 컷, EOS 무시 | 오라클 길이 비교 |

모델별로 두 모드 모두 보고. 비교 행 안에서 모드 혼합 금지 (free vs free, gt_length vs gt_length만).

---

## 7. 사전학습 산출물 (Hugging Face)

| 런 | 레포 |
|---|---|
| Qwen2.5-7B Full FT 100k (ckpt-7500) | https://huggingface.co/sungkwang2/qwen2.5-7b-en-fi-fullft-100k |
| Fast-dLLM v2 7B Full FT 100k | https://huggingface.co/sungkwang2/fastdllm-v2-7b-en-fi-fullft-100k |
| Fast-dLLM v2 7B LoRA r=256 100k | https://huggingface.co/sungkwang2/fastdllm-v2-7b-en-fi-lora256-100k |

HF id를 직접 평가에 사용 가능 (Fast-dLLM Full FT 예시):
```bash
python3 eval/eval_fastdllm.py --ckpt sungkwang2/fastdllm-v2-7b-en-fi-fullft-100k \
  --mode free --out outputs/<run>/eval_free.json
```

---

## 8. 로드맵

- [x] Fast-dLLM v2 Full FT 100k × 1 ep (free)
- [x] Fast-dLLM v2 LoRA r=256 100k × 1 ep (free)
- [x] Qwen Full FT 100k × 1 ep (75%, ckpt-7500, free)
- [ ] Qwen LoRA r=256 100k × 1 ep
- [ ] 모든 행에 대해 `gt_length` 재측정
- [ ] 에러 분석 (≥ 3 축, `reports/REPORT_GUIDE.md` 참조)

---

## 9. 문제 해결

| 증상 | 해결 |
|---|---|
| ROPE에서 `KeyError: 'default'` | transformers 4.57.6 확인 |
| `'DynamicCache' object has no attribute 'key_cache'` | Q3 패치 확인 |
| `SFTTrainer ... unexpected keyword 'tokenizer'` | Q5 패치 확인 |
| Fast-dLLM 학습 loss가 step 1부터 NaN | Q2 (`gradient_checkpointing=False`) 확인 |
| `batch_sample` 안에서 크래시 (`past_key_values is None`) | Q4 가드 확인 |
| Fast-dLLM Full FT CUDA OOM | `--grad_accum 16`; 또는 LoRA 전환 |
| 학습 중 디스크 부족 | `--save_steps` 키우기; 기본 4000 (10k step당 2 ckpt) + `save_total_limit=2` 적용됨 |

---

## 참고

- Fast-dLLM v2 7B — https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B
- Qwen2.5-7B-Instruct — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- OPUS-100 — https://huggingface.co/datasets/Helsinki-NLP/opus-100
- Reference NMT — https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-fi
