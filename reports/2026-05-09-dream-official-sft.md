# 실험일지: Dream-7B 공식 SFT 포팅 및 재실험

> 날짜: 2026-05-09

---

## 배경

이전 Dream-7B 번역 실험(v1)에서 Qwen 대비 극히 낮은 성능(chrF 7.03 vs 33.61)을 보였다.
원인을 파악하기 위해 Dream 공식 레포([DreamLM/Dream](https://github.com/DreamLM/Dream))의 SFT 학습 코드를 분석하였다.

## 원인 분석

공식 코드(`src/trainer/fsdp_sft_trainer.py`, `src/diffllm/gen_utils.py`)와 이전 실험 코드를 비교한 결과, **5가지 근본적 차이**를 발견:

1. **Logit shift 누락**: Dream은 자기 위치 토큰을 예측(`cat([logits[:,0:1], logits[:,:-1]])`). 이전 코드는 AR 방식(next-token)으로 엉뚱한 loss 학습.
2. **마스킹 범위 오류**: 이전 코드는 prompt까지 마스킹 → 모델이 "무엇을 번역해야 하는지"를 모르는 상태에서 학습. 공식 코드는 response-only.
3. **Attention mask**: 공식 코드는 4D bidirectional, 이전 코드는 2D(causal 적용 가능성).
4. **Time reweighting 부재**: 공식 코드는 `1/t`로 저마스크 비율에 높은 가중치. 이전 코드는 균일.
5. **LR 20배 과다**: 이전 2e-4, 공식 기본값 1e-5.

## 수행 작업

### 1. 공식 SFT 코드 분석 (DreamLM/Dream)

- `git clone` 후 전체 구조 분석
- 핵심 파일: `gen_utils.py` (q_sample), `fsdp_sft_trainer.py` (loss), `sft_dataset.py` (데이터)
- 공식 config 확인: `sft_trainer.yaml` (lr=1e-5, time_reweighting=original, token_reweighting=false)

### 2. 단일 GPU 포팅 (`train_dream_official.py`)

- FSDP/verl/hydra 의존성 제거
- `q_sample()` 그대로 복사
- `_compute_loss_and_backward()` → `compute_dream_loss()` (FSDP 제거, 단일 GPU)
- `SFTDataset` → `DreamSFTDataset` (JSONL 입력, chat_template 적용)
- argparse CLI로 공식 config 기본값 반영

### 3. 환경 이슈 해결

| 이슈 | 해결 |
|------|------|
| `ROPE_INIT_FUNCTIONS['default']` KeyError (transformers 5.0) | transformers 4.46.2로 다운그레이드 |
| torchao 0.10.0 < 0.16.0 (peft 요구) | torchao 0.17.0 설치 |

### 4. 1k 데이터 학습

```
python3 train_dream_official.py \
  --train_file data/train_1k.jsonl \
  --val_file data/val_1k.jsonl \
  --epochs 3
```

결과:
- 학습 시간: **2.9분**
- Loss 추이: train 0.16→0.17→0.18 / val 0.13→0.18→0.16
- Best val loss: **0.1308** (Epoch 1)
- 이전 v1의 loss 5.52와는 완전히 다른 규모

### 5. 평가

| 모델 | chrF | BLEU | 데이터 |
|------|:----:|:----:|:------:|
| opus-mt (Reference) | 56.91 | 37.45 | — |
| Qwen + LoRA | 33.61 | 16.27 | 1k |
| **Dream v2 (공식 SFT)** | **19.51** | **9.79** | **1k** |
| Dream v1 (이전) | 7.03 | 0.83 | 10k |

## 결론

1. **공식 SFT 방식 적용으로 chrF 7→19.5 (2.8배), BLEU 0.8→9.8 (11.8배) 개선.**
2. 10k→1k로 데이터를 줄이고도 성능이 대폭 향상 — **학습 방식의 정확성 > 데이터 양**.
3. **Qwen(33.6) 대비 아직 14pt 뒤진다.** 번역은 순차적 생성이 유리한 태스크로, AR 모델이 구조적 우위. Dream의 강점은 planning 태스크(Sudoku 81 vs Qwen 21).
4. 추가 개선 가능성: 10k 데이터 적용, epoch 증가, token_reweighting 활성화, CART reweighting 실험.

## 생성 파일

- `train_dream_official.py` — Dream 공식 SFT 단일 GPU 포팅
- `eval_dream_official.py` — 평가 스크립트
- `outputs/dream_official/` — adapter, loss curve, eval results, training meta
