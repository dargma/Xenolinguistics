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

| 모델 | 방식 | 학습 진행 | 체크포인트 | chrF (free) | BLEU (free) | chrF (gt_length) | BLEU (gt_length) |
|---|---|:---:|---|:---:|:---:|:---:|:---:|
| `opus-mt-tc-big-en-fi` | NMT 베이스라인 | — | [HF](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-fi) | **56.91** | **37.45** | — | — |
| Qwen2.5-7B | Full FT, lr=2e-5 | 1 ep, 7500/10000 (75%) | [HF](https://huggingface.co/sungkwang2/qwen2.5-7b-en-fi-fullft-100k) | **47.72** | **27.65** | **46.33** | **25.18** |
| Fast-dLLM v2 7B | Full FT, lr=2e-5 | 1 ep, 10000/10000 (100%) | [HF](https://huggingface.co/sungkwang2/fastdllm-v2-7b-en-fi-fullft-100k) | 40.86 | 17.22 | 40.25 | 18.79 |

**평가 방법** (두 모드 모두):
- `mnt = max_new_tokens = 256` 동일 (Diffusion의 denoise budget 비대칭 제거).
- 모델은 256 토큰까지 생성 → `gen_ids[input_prompt:]` 만 남김.
- **`free`**: 첫 EOS 토큰에서 컷. EOS 없으면 256까지 keep.
- **`gt_length`**: 먼저 EOS 컷 → 그 후 `len_tok(fi_ref)` 토큰 cap (EOS와 ref 길이 중 빠른 것).
- 디코드는 `skip_special_tokens=True`. chrF/BLEU는 `sacrebleu` 기본값 (corpus 레벨, n=100).
- Reference NMT는 beam=4, `eval/eval_reference.py`.

---

## 1b. 결과 — 인공어 (conlang) en→인공어, full-FT

가설: 인공어는 어휘+어순이 영어와 다르고 사전학습에 (대체로) 없으므로, 양방향 디코딩의
Diffusion LLM이 AR LLM보다 — 특히 어순을 — 더 잘 학습할 것이다 (H1).

### Klingon (OPUS Tatoeba en-tlh, 12.7k, OVS 어순)
양쪽 동일: full-FT, lr 2e-5, 1 ep, eff.batch 8. eval: 0-shot, **양쪽 free-length 대칭**,
n=300 (`eval_ft.py`). 숫자는 `outputs/klingon_ft_eval_{ar,diffusion}.json`.

| 모델 | 체크포인트 | chrF | BLEU | EM | 어순 τ |
|---|---|:---:|:---:|:---:|:---:|
| Qwen2.5-7B (AR) | [HF](https://huggingface.co/sungkwang2/klingon-en2tlh-qwen2.5-7b-fullft) | **38.43** | **10.40** | 8.33 | 0.909 |
| Fast-dLLM v2 7B (Diffusion) | [HF](https://huggingface.co/sungkwang2/klingon-en2tlh-fastdllm-v2-7b-fullft) | 35.67 | 2.55 | **8.67** | **0.936** |

해석: 혼합/메트릭 의존적. **어순 τ에서 Diffusion>AR (0.936 vs 0.909) → H1 약한 지지**;
전체 유창성(chrF/BLEU)은 AR 우세 (Diffusion 일부 입력 degenerate). 상세·loss 곡선·caveat:
[`reports/2026-05-28-klingon-ar-vs-diffusion.md`](reports/2026-05-28-klingon-ar-vs-diffusion.md).
Khalani(55쌍)는 FT엔 너무 작아 ICL/탐색용으로만 취급.

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


## 3. 데이터 준비

```bash
python3 dataset/prepare_dataset.py --sizes 1k,10k,100k
```
`dataset/data/{train,val,test}_{1k,10k,100k}.jsonl` 를 생성. 각 줄 스키마:
```json
{"instruction": "Translate to Finnish: <EN>", "output": "<FI>", "en": "<EN>", "fi": "<FI>"}
```
다른 언어쌍 → `dataset/prepare_dataset.py` 상단 상수만 수정. 스크립트 포크 금지.

---

## 4. 학습 + 평가

런 하나 = `outputs/<run>/` 디렉터리 하나 = `eval_*.json` (모드별) 하나.

### 4.1) Reference NMT 베이스라인 (학습 없음, < 1분)
```bash
python3 eval/eval_reference.py
```

### 4.2) Qwen2.5-7B Full FT (≈ 3시간)
```bash
python3 train/train_qwen_v4.py \
  --train_file dataset/data/train_100k.jsonl --val_file dataset/data/val_1k.jsonl \
  --output_dir outputs/qwen_100k_fullft \
  --lora_rank 0 \
  --lr 2e-5 --epochs 1
python3 eval/eval_qwen.py --adapter outputs/qwen_100k_fullft/final \
  --mode free      --out outputs/qwen_100k_fullft/eval_free.json
python3 eval/eval_qwen.py --adapter outputs/qwen_100k_fullft/final \
  --mode gt_length --out outputs/qwen_100k_fullft/eval_gt_length.json
```

### 4.3) Fast-dLLM v2 7B — Full FT (≈ 3시간)
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

## 5. 평가 모드

두 모드 모두 `max_new_tokens=256` (동일 생성 예산 → Diffusion 모델의 denoise budget 비대칭 제거). 차이는 후처리만:

| 모드 | 후처리 | 용도 |
|---|---|---|
| `free` | 첫 EOS에서 컷 | 자연 길이 비교 |
| `gt_length` | EOS 스트립 후 `len_tok(fi_ref)` 토큰만큼 컷 | 오라클 길이 비교 |

## 참고

- Fast-dLLM v2 7B — https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B
- Qwen2.5-7B-Instruct — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- OPUS-100 — https://huggingface.co/datasets/Helsinki-NLP/opus-100
