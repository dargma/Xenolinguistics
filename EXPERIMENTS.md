# Dream 7B en→fi 실험 목록

> **목표**: Dream 7B을 영어→핀란드어 번역에 LLaDA 8B 수준으로 학습시키기
> **현재 격차**: Dream chrF ~22 (10K free-length) vs LLaDA chrF 38.73 (10K GT-length)
> **데이터**: OPUS-100 en-fi, 평가는 DiffusionLLM-style oracle GT length

---

## 결과 요약 (chrF, free-length 기준 별도 표시)

| 실험 | data | LR | weight | LoRA | epoch | chrF | BLEU | 상태 |
|------|------|----|----|------|-------|------|------|------|
| dream_filtered_lr5e5_ep5 | 10K filt | 5e-5 | 1/t | q/k/v/o | 5 | **21.97** | 8.58 | done (Dream best) |
| dream_v4_kd_1k | 1K KD | 2e-6 | CART | all-linear | 3 | **22.01** | 15.98 | done |
| dream_filtered_lr5e5 | 10K filt | 5e-5 | 1/t | q/k/v/o | 3 | 21.76 | 10.88 | done |
| dream_10k_filtered | 10K filt | 1e-5 | 1/t | q/k/v/o | 3 | 21.79 | 10.16 | done |
| dream_filtered_lr2e5_ep5 | 10K filt | 2e-5 | 1/t | q/k/v/o | 5 | 20.27 | 7.42 | done |
| dream_official_10k | 10K | 1e-5 | 1/t | q/k/v/o | 3 | 18.56 (19.84 GT) | 9.58 | done |
| dream_v4_exp1_10k | 10K | 2e-6 | CART | all-linear | 3 | 16.02 | 15.82 | done |
| **dream_lr5e5_100k_ep1** | **100K** | 5e-5 | 1/t | q/k/v/o | 1 | TBD | TBD | **running** |

비교 baseline:
- **opus-mt** (Helsinki-NLP, 250M, dedicated NMT): chrF 56.91
- **Qwen 2.5 7B + LoRA** (AR baseline, 10K): chrF 41.72
- **LLaDA 8B + LoRA** (Diffusion 비교, 10K, GT-length): chrF **38.73**

---

## 가설 & 확인 사항

### H1: spatial loss weight 제거 (LLaDA 따라가기)

**Dream CART**의 spatial 가중치 $w(t,x_t,n) = \frac{1}{2}\sum_i \mathbf{1}[x_t^i \neq M] \cdot \text{Geo}(p, |n-i|-1)$는 번역에 비최적일 가능성. target(핀란드어) 전체 마스킹된 상태에선 unmasked prompt(영어)와의 거리만 가중치 좌우 → 위치별 학습 불균등.

LLaDA는 모든 마스크 토큰에 균등 가중 ($1/t$). 이걸 Dream에 적용:

- **시도 1 (1/t LLaDA-style)**: 분산 폭발. lr=2e-6 + all-linear에서 loss 1.3~2.8 진동. → **실패**, all-linear와 1/t 조합이 LoRA 표면적 너무 커서 variance 흡수 불가.
- **시도 2 (constant=1, 시간/위치 둘 다 무관)**: loss 0.3~1.4 큰 진동. step 800에서 baseline과 TIE 보였으나 분산 매우 큼. → **불안정**, eval까지 가도 baseline과 비슷할 가능성.

**결론**: LLaDA loss 그대로 옮기는 건 우리 LoRA + 작은 batch 환경에 부적합. Dream 자체 best recipe 찾는 게 우선.

### H2: 데이터 스케일링 (10K → 100K)

기존 모든 Dream 실험이 chrF ~22 천장. 데이터 부족이 원인일 가능성.

- 메모리: "Data scaling STILL fails: 10k worse than 1k with v4 config" — 하지만 그건 CART + lr=2e-6 + all-linear의 문제로 보임.
- 검증된 best recipe (1/t + q/k/v/o + lr=5e-5)로 100K 스케일링 중. → **Phase 1c 진행 중**.

### H3: Dream-official perbatch_cutoff augmentation 누락

Dream 공식 SFT 스크립트(`run_sft_tulu3.sh`)에 명시된 핵심 augmentation. batch별 response 길이를 batch 내 랜덤 선택값으로 truncation → diffusion의 "가변 출력 길이" 학습.

- **시도 (lr=5e-5 + 100K + perbatch_cutoff)**: valid_tok=15, loss=5.5 폭발. 우리 데이터(opus-100 단문)에 적용 시 너무 짧게 자름. batch_size=8 + 긴 sequence(Tulu3) 환경 가정한 augmentation. → **실패**, 단문 번역엔 부적합.

### H4: Pretraining 다국어 부재 (수정 불가)

- Dream: Dolma v1.7 (영어 only) + Qwen2.5 init (영어/중국어 중심) — 580B 토큰
- LLaDA: 2.3T 토큰 multilingual (CMMLU 69.9 vs LLaMA3 50.7) — 다국어 명시
- 이건 fine-tune으로 못 고침. **Dream 본질적 천장**.

---

## 실행 중 / 대기 중 실험

### Phase 1c (running): 100K + lr5e-5 + 1/t + q/k/v/o (no perbatch_cutoff)

- Best 10K recipe scale-up. 1 epoch.
- 중간 체크포인트 step 2500, 5000, 7500, 10000 → 도중 eval 가능.
- 성공 기준: chrF > 22 (천장 돌파). 스트레치: chrF > 30.

### Phase 2a (pending): KD-augmented 100K

- Qwen2.5 7B로 EN→FI 50K 의역 생성 → 50K gold + 50K KD 혼합
- dream_v4_kd_1k에서 chrF 22.01 (+0.04 over baseline) 효과 확인
- 100K 스케일에서 효과 클 것으로 예상

### Phase 2b (pending): DoRA variant

- peft `use_dora=True` — magnitude × direction 분리, LoRA 표현력 ↑
- 같은 파라미터 수로 더 강한 표현 학습

---

## 평가 프로토콜

DiffusionLLM (Ye et al. 2023) oracle length 방식:

```
input = [prompt(chat-template)] + [MASK × gt_len] + [EOS]
```

- prompt와 trailing EOS는 **고정** (denoise 안 함)
- mask block만 iterative denoising (confidence-based unmasking, 64 steps)
- gt_len = `len(tokenizer.encode(fi_reference))`

스크립트: `eval_dream_gt.py --adapter_path X --n_eval 100`

---

## 환경 요구사항

- `transformers==4.46.2` (5.x는 Dream의 `ROPE_INIT_FUNCTIONS` 깨짐)
- `torchao>=0.17.0` (peft all-linear 호환)
- `trl==0.12.0`
- `peft`, `sacrebleu`, `bitsandbytes`
- Dream 캐시 클리어: `rm -rf /root/.cache/huggingface/modules/transformers_modules/Dream*` 필요 시
