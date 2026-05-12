# Dream-7B 근본 원인 분석 및 개선 실험 계획

> 2026-05-12 | 목표: Dream chrF 21.79 -> 35+ (Qwen 수준 접근)

## 진단 요약

### A. 공식 코드 대비 설정 오류 (코드 비교로 확인)

| # | 문제 | 현재 | 공식/권장 | 예상 영향 |
|---|------|------|----------|----------|
| 1 | LoRA target_modules | q/k/v/o_proj only | **all-linear** (MLP+lm_head 포함) | +++ |
| 2 | Learning Rate | 1e-5 | **2e-6** (공식 run script) | ++ |
| 3 | Time reweighting | "original" (1/t) | **"cart"** (context-adaptive) | ++ |
| 4 | max_length padding | 512 (90.5% waste) | **128** (covers 99%) | + |

### B. 문헌 조사 결과 (구조적 한계 + 개선 기법)

번역에서 AR을 이긴 유일한 diffusion 모델 = **Difformer** (Enc-Dec, NAACL 2024, BLEU +0.57)
- 성공한 diffusion 번역 모델은 **전부 encoder-decoder**. Decoder-only는 없음.
- Dream은 decoder-only → 번역에 구조적 불리

적용 가능한 기법:

| 기법 | 출처 | 재학습 | 기대 효과 |
|------|------|--------|----------|
| MBR Decoding (N=10) | CDCD | 불필요 | +2~5 chrF |
| Low-confidence remasking | Dream 내장 | 불필요 | +1~3 chrF |
| Length prediction | Difformer | 소규모 | gap 해소 |
| Knowledge Distillation (Qwen 출력) | EACL 2024 bench | 재학습 | +5~10 chrF |

### C. 정상 확인 항목

- Logit shift, q_sample, 4D attention, loss_mask: 공식 코드와 일치
- Eval 코드: diffusion_generate 내부에서 shift 처리, 정상

---

## 실험 설계 원칙

- **한 번에 하나의 변수만 변경** (ablation)
- 단, Exp-1은 공식 설정 재현이므로 복수 변경 허용
- 모든 실험은 **10k 데이터, test_1k 100문장** 기준
- baseline: Dream 10k 원본 chrF=18.56 / Dream 10k filtered chrF=21.79

---

## Phase A: 학습 설정 수정 (Exp 1-4)

### Exp-1: 공식 설정 완전 재현 (all-linear + lr2e-6 + cart + max128)

**가설**: 공식 SFT 설정과의 3가지 핵심 차이를 동시에 수정하면 성능이 크게 개선
**변경**: target_modules=all-linear, lr=2e-6, time_reweighting=cart, max_length=128
**판단 기준**:
- chrF 30+ → 설정 차이가 근본 원인. Phase A ablation 진행
- chrF 25~30 → 부분 개선. Phase B + C로 추가 개선
- chrF ~20 → 설정 아닌 구조적 한계. Phase C 집중

### Exp-2: all-linear 단독 (lr=1e-5, rw=original, max=512)
### Exp-3: lr=2e-6 단독 (modules=qkvo, rw=original, max=512)
### Exp-4: cart 단독 (modules=qkvo, lr=1e-5, max=512)

---

## Phase B: 최적화 (Exp-1 결과 기반)

### Exp-5: max_length=256 (Exp-1 기반, 128 vs 256 비교)
### Exp-6: LoRA r=64, alpha=128 (Exp-1 기반, 용량 증가)
### Exp-7: Filtered data (Exp-1 기반, >=5 words)

---

## Phase C: 문헌 기반 추론 개선 (재학습 불필요)

### Exp-8: MBR Decoding

**가설**: N=10 후보 생성 후 chrF 기반 선택 → 번역 품질 향상
**구현**:
- temp=0.4~0.6으로 N=10 후보 생성
- pairwise chrF 계산, 평균 chrF 최고인 후보 선택
- **기존 adapter 재사용** (Exp-1 best or baseline)

### Exp-9: Remasking 전략 비교

**가설**: entropy/maskgit_plus/topk_margin 중 번역에 최적인 전략이 다를 수 있음
- Dream의 diffusion_generate에 이미 구현되어 있음
- 기존 adapter로 alg 변경만으로 실험

### Exp-10: Length Prediction (GT length 활용)

**가설**: free_length(64고정) vs gt_length(정답길이) 격차 측정 → 길이 예측 가치 확인
- 먼저 gt_length로 상한 측정
- 차이 크면 → source 길이 * 계수로 간단한 length heuristic 적용

---

## Phase D: 재학습 기반 개선 (선택)

### Exp-11: Knowledge Distillation

**가설**: Qwen 10k 출력을 pseudo-reference로 사용하면 Dream의 multimodality 문제 완화
**구현**:
1. Qwen adapter로 train_10k의 모든 en에 대해 fi 번역 생성
2. 이 번역을 output으로 사용한 새 JSONL 생성
3. Dream을 이 데이터로 SFT

---

## 실행 순서

```
1. Exp-1 (공식 설정 재현) — 최우선
2. Exp-1 결과에 따라:
   - 큰 개선 → Exp 2,3,4 ablation → Exp 5,6,7 최적화
   - 작은 개선 → Exp 8,9,10 추론 개선 병행
3. Exp-8 (MBR) — 재학습 불필요, 언제든 실행 가능
4. Phase D — Phase A~C 결과 종합 후 결정
```

## 스크립트

- 학습: `train_dream_v4.py` (target_modules, lr, reweighting 설정 가능)
- 평가: `eval_dream_v4.py` (adapter_path 인자)
- 실행: `bash run_ablation.sh [1-7]`

## 평가 체크리스트

각 실험마다 확인:
- [ ] chrF, BLEU (test_1k 100문장)
- [ ] 정성 비교 (동일 5문장)
- [ ] val_loss 추이 (수렴 여부)
- [ ] trainable params 수
- [ ] 학습 시간
- [ ] diffusion_generate 정상 동작 여부
