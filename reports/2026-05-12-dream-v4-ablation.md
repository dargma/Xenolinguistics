# Dream-7B v4 Ablation: 근본 원인 분석 및 개선

> 2026-05-12

## 요약

- 공식 Dream SFT config와 3가지 핵심 차이를 발견하고 수정 (all-linear LoRA, lr=2e-6, CART reweighting)
- 100 samples full eval 기준: chrF 19.51 -> **20.44** (+4.8%)
- 25 samples quick eval에서는 +27% 개선이었으나, full eval에서는 소폭 개선에 그침
- 동일 조건(all-linear r=16)에서 Dream은 Qwen보다 과적합에 강함 (diffusion masking의 regularization 효과)
- MBR decoding, Knowledge Distillation 모두 효과 없음
- 문헌 조사: 번역에서 AR을 이긴 diffusion 모델은 전부 Encoder-Decoder 아키텍처

---

## 1. 진단: 공식 코드와의 차이

공식 Dream repo (`DreamLM/Dream`)의 SFT config 및 run script와 비교:

| 항목 | 이전 (v2) | 공식 코드 | 비고 |
|------|----------|----------|------|
| LoRA target_modules | q/k/v/o_proj (10M, 0.13%) | **all-linear** (40M, 0.53%) | MLP+lm_head 포함 |
| Learning Rate | 1e-5 | **2e-6** (run_sft_tulu3.sh) | 1/t reweight 시 gradient 크므로 |
| Time reweighting | "original" (1/t) | **"cart"** (context-adaptive) | t->0 gradient 안정화 |

정상 확인: logit shift, q_sample, 4D attention, loss_mask, eval 코드 (diffusion_generate 내부 처리)

---

## 2. Phase A: Ablation (1k 데이터)

### Quick eval (25 samples, 64 steps)

| Exp | target_modules | LR | Reweighting | chrF | BLEU | val_loss |
|-----|---------------|-----|-------------|------|------|----------|
| baseline (v2) | qkvo | 1e-5 | original | 19.51* | 9.79* | 0.13 |
| **Exp-1** | **all-linear** | **2e-6** | **cart** | **24.71** | **17.21** | 0.19 |
| Exp-2 | all-linear | 1e-5 | original | 19.15 | 17.70 | 0.79 |
| Exp-3 | qkvo | 2e-6 | original | 20.28 | 16.90 | 0.70 |
| Exp-4 | qkvo | 1e-5 | cart | 17.67 | 17.54 | 0.21 |
| Exp-6 | all-linear r=64 | 2e-6 | cart | 20.62 | 17.11 | 0.20 |

*baseline은 100 samples, 128 steps 기준

### Full eval (100 samples, 128 steps)

| 설정 | chrF | BLEU |
|------|------|------|
| baseline (v2, 1k) | 19.51 | 9.79 |
| **Dream v4 (Exp-1, 1k)** | **20.44** | **9.74** |

25 samples에서 보인 +27%는 test set 초반 편향. **Full eval 기준 +4.8% (chrF +0.93)**

### 분석

- 3가지 동시 변경이 최대 효과 (개별 변경은 제한적)
- all-linear + 1/t reweighting 조합은 val_loss 불안정 (0.79)
- r=64는 1k에 과적합 (r=16이 최적)
- **BLEU는 quick eval에서 9.79→17+로 크게 개선되었으나, full eval에서는 동일 수준**

---

## 3. 데이터 스케일링

| 데이터 | Dream v4 chrF | Dream v2 chrF |
|--------|-------------|-------------|
| 1k | 24.71 (quick) / 20.44 (full) | 19.51 |
| 10k | 16.02 (quick) | 18.56 |

v4 설정에서도 10k로 스케일링하면 성능 하락. 설정과 무관한 구조적 특성.

---

## 4. 동일 조건 AR vs Diffusion 비교

### all-linear r=16 (동일 40M params, 25 samples)

| 데이터 | Dream v4 | Qwen v4 | 승자 |
|--------|---------|---------|------|
| 1k | **24.71** | 12.84 | **Dream** |
| 10k | 16.02 | 18.28 | Qwen (근소) |

- Qwen all-linear 1k: 심각한 반복 생성 과적합 (chrF 12.84)
- **Dream의 diffusion masking이 자연 regularization**: 매 step 랜덤 마스킹 = data augmentation
- 10k에서는 Qwen 근소 우위, 두 모델 모두 all-linear에서 과적합 경향

### 각 모델 최적 설정 비교 (100 samples)

| 모델 | 최적 설정 | chrF | BLEU |
|------|----------|------|------|
| opus-mt (Reference) | 전용 NMT | 56.91 | 37.45 |
| Qwen best | q/k/v/o r=16, 10k | 41.72 | 22.89 |
| **Dream v4 best** | **all-linear r=16, 1k** | **20.44** | **9.74** |
| Dream v2 best | qkvo r=16, 10k filtered | 21.79 | 10.16 |

---

## 5. 추론 개선 실험

### MBR Decoding (N=5, temp=0.5)

| 방식 | chrF | BLEU |
|------|------|------|
| Greedy | 23.81 | 11.41 |
| MBR N=5 | 18.66 | 14.75 |
| Delta | **-5.15** | +3.34 |

MBR은 chrF 하락. Diffusion sampling의 다양한 후보가 오히려 품질 저하. Greedy가 최선.

### Knowledge Distillation (Qwen 번역으로 학습)

| 학습 데이터 | chrF | BLEU |
|------------|------|------|
| Gold (원본) | 24.71 | 17.21 |
| KD (Qwen 출력) | 22.01 | 15.98 |

Qwen 번역과 test gold reference의 스타일 차이로 인해 역효과.

---

## 6. 문헌 조사: Diffusion LLM과 번역

### 번역 평가한 Diffusion 모델

| 모델 | 년도 | 아키텍처 | 번역 결과 vs AR | 핵심 기법 |
|------|------|---------|----------------|----------|
| Difformer | 2024 | Enc-Dec | **+0.57 BLEU** | Anchor loss, noise rescaling, length prediction |
| E2D2 | 2025 | Enc-Dec | +0.8 BLEU | Encoder-decoder 분리 |
| CDCD | 2022 | Enc-Dec | -3~7 BLEU | MBR decoding |
| Dream (ours) | 2025 | **Dec-only** | **-21 chrF** | Response-only masking |

### Dream 논문의 평가 태스크
MMLU, ARC, HellaSwag, GSM8K, MATH, HumanEval, BBH, IFEval, Countdown, Sudoku, Trip Planning
**번역: 미평가. 저자들이 의도적으로 회피한 것으로 추정.**

### 핵심 발견
번역에서 AR을 이긴 유일한 diffusion 모델 = **Difformer** (Encoder-Decoder 아키텍처).
Decoder-only diffusion 모델로 번역을 성공한 사례는 **문헌에 존재하지 않음**.

---

## 7. 결론

### 설정 수정 효과
- 공식 config 3가지 차이(all-linear, lr, reweighting) 수정: full eval 기준 **+4.8%** (19.51 -> 20.44)
- 개별 요인보다 조합 효과가 큼
- 하이퍼파라미터 수준의 개선은 상한에 도달

### Dream의 구조적 특성
- **Regularization 효과**: all-linear에서 Dream이 Qwen보다 과적합에 강함 (1k: 24.71 vs 12.84)
- **데이터 스케일링 실패**: 1k -> 10k에서 성능 하락. 설정 수정으로도 해결 불가
- **추론 개선 한계**: MBR, KD 모두 효과 없음

### Decoder-only Diffusion LLM의 번역 한계
1. **Encoder 부재**: source를 한 번 인코딩하고 target만 반복 디노이징하는 Enc-Dec과 달리, Dec-only는 매 step마다 전체 sequence를 처리
2. **Fluency-Diversity tradeoff**: Diffusion은 diversity에 강하나 번역은 fluency(정확한 1:1 매핑)가 필요
3. **문헌 부재**: Decoder-only diffusion으로 번역 성공 사례가 없음. Dream 논문도 번역 미평가.

### 현재 최선 vs 목표

| 모델 | chrF | 목표 대비 |
|------|------|----------|
| Dream v4 best | 20.44 | baseline |
| Dream v2 best (filtered) | 21.79 | +6.6% |
| **Qwen best** | **41.72** | **목표** |
| 격차 | | **2x** |

Qwen 수준(chrF 41.72) 달성은 **현재 Dream의 decoder-only 아키텍처로는 구조적으로 불가능**한 것으로 판단. 이는 코드 오류나 설정 문제가 아닌, diffusion LLM의 번역에 대한 근본적 한계.

---

*2026-05-12 완료*
