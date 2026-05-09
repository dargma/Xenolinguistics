# 실험 보고서: Dream-7B 공식 SFT 포팅 및 10k 비교

> 날짜: 2026-05-09

---

## 1. 환경

| 항목 | 값 | 비고 |
|------|-----|------|
| GPU | NVIDIA RTX PRO 6000 Blackwell | 98GB VRAM |
| PyTorch | 2.10.0+cu128 | |
| **Transformers** | **4.46.2** | Dream 호환 필수. 5.x에서 ROPE_INIT_FUNCTIONS KeyError 발생 |
| PEFT | 0.19.1 | |
| TRL | 1.4.0 | Qwen SFT용. `processing_class` API 사용 |
| torchao | 0.17.0 | peft 0.19.1이 >=0.16.0 요구 |
| sacrebleu | 2.5+ | chrF/BLEU 평가 |

**환경 이슈 해결**:
- `ROPE_INIT_FUNCTIONS['default']` KeyError → transformers 4.46.2 다운그레이드 + HF 캐시 삭제
- torchao 0.10.0 호환 오류 → 0.17.0 업그레이드

---

## 2. Dream v2: 공식 SFT 포팅

### 배경

이전 Dream v1 학습(chrF 7.03)은 공식 학습 방식과 5가지 근본적 차이가 있었다. [DreamLM/Dream](https://github.com/DreamLM/Dream)의 `src/trainer/fsdp_sft_trainer.py`를 분석하여 단일 GPU로 포팅하였다.

### 핵심 수정사항

| 항목 | v1 (잘못됨) | v2 (공식 포팅) |
|------|------------|---------------|
| Logit shift | 없음 (AR 방식) | `cat([logits[:,0:1], logits[:,:-1]])` — 자기 위치 예측 |
| 마스킹 범위 | 전체 시퀀스 랜덤 | response-only (`loss_mask`로 prompt 제외) |
| Attention mask | 2D (causal 가능) | 4D bidirectional `[B,1,L,L]` |
| Time reweighting | 없음 | `1/t` (공식 `original` 모드) |
| 마스킹 함수 | 단순 랜덤 | `q_sample()` (EOS 그룹 처리 포함) |
| LR | 2e-4 | 1e-5 (공식 기본값) |

### 1k 학습 검증 결과

| 지표 | v1 (10k 데이터) | v2 (1k 데이터) | 개선 |
|------|:-:|:-:|------|
| chrF | 7.03 | **19.51** | +12.48 (2.8x) |
| BLEU | 0.83 | **9.79** | +8.96 (11.8x) |

데이터를 10배 줄이고도 성능 2.8배 향상 — **학습 방식의 정확성 > 데이터 양**.

---

## 3. Qwen vs Dream 10k 비교

### 3-1. 데이터셋

| 항목 | 내용 |
|------|------|
| 출처 | Helsinki-NLP/opus-100 (en-fi) |
| 내용 | 영화 자막 + EU 공식문서 영어-핀란드어 병렬 코퍼스 |
| 형식 | JSONL: `{"instruction": "Translate to Finnish: ...", "output": "...", "en": "...", "fi": "..."}` |

| 셋 | Train | Val | Test | 인덱스 범위 |
|----|------:|----:|-----:|------------|
| 1k | 800 | 100 | 100 | opus-100 0~999 |
| 10k | 8,000 | 1,000 | 1,000 | opus-100 0~9,999 |
| 100k | 80,000 | 10,000 | 10,000 | *TBD* |

> **주의**: 1k test(인덱스 900-999)와 10k test(인덱스 9000-9999)는 **서로 다른 문장**이다. 스케일 간 공정 비교 시 동일 test set 사용 필요.

### 3-2. 학습 하이퍼파라미터

| 항목 | Qwen2.5-7B (AR) | Dream-7B v2 (Diffusion) |
|------|:---:|:---:|
| Base model | Qwen/Qwen2.5-7B-Instruct | Dream-org/Dream-v0-Instruct-7B |
| LoRA r / alpha | 16 / 32 | 16 / 32 |
| Target modules | q,k,v,o_proj | q,k,v,o_proj |
| Trainable params | 10M (0.13%) | 10M (0.13%) |
| Learning rate | 2e-4 | 1e-5 |
| LR scheduler | Cosine | Cosine |
| Optimizer | AdamW | AdamW (beta=0.9,0.95, wd=0.01) |
| Epochs | 3 | 3 |
| Batch size | 4 | 8 |
| Grad accumulation | 2 (effective=8) | 1 (effective=8) |
| Loss 방식 | Standard CE (next-token) | CE(masked) / t reweighting |
| Attention | Causal (triangular) | Bidirectional (4D full) |
| 학습 시간 | 13.3분 | *진행 중* |

**LR 차이 설명**: Qwen은 AR의 표준 SFT LR(2e-4). Dream은 공식 레포 기본값(1e-5)으로, diffusion loss의 time reweighting으로 인해 gradient 스케일이 다르기 때문.

### 3-3. Loss 커브 (Y축/X축 동일 스케일)

> 1k 학습 기준. 10k 학습 완료 시 업데이트 예정.

| Qwen (AR) | Dream v2 (Diffusion) |
|:---:|:---:|
| ![Qwen](fig-qwen-vs-dream-1k-loss.png) | *(동일 그래프 내 우측 패널)* |

- **Qwen**: 초기 loss 4.5 → 최종 1.1. 전형적 AR SFT 수렴.
- **Dream v2**: 0.04~0.5 변동, 평균 0.15 수준 유지. `1/t` reweighting으로 loss 스케일이 AR과 다르며, timestep 샘플링에 의한 변동은 정상.

**해석**: 두 모델의 loss는 직접 비교 불가. Qwen은 next-token CE, Dream은 masked-token CE / t. 수렴 여부는 각각의 추이로 판단.

### 3-4. 벤치마크 평가

**평가 지표**:
- **chrF**: Character n-gram F-score. 형태론 복잡 언어에서 BLEU보다 신뢰도 높음. (0~100)
- **BLEU**: Word n-gram precision. 전통적 MT 지표. (0~100)
- **Reference NMT**: OPUS-100 en-fi 전용 학습 모델. "이 정도면 합리적" 기준선.
- **Test set**: 동일 test_1k (100문장, 인덱스 900-999) 기준. 스케일 간 공정 비교.

#### 데이터 스케일링에 따른 성능 추이

| 모델 | 1k | | 10k | | 100k | |
|------|:----:|:----:|:----:|:----:|:----:|:----:|
| | chrF | BLEU | chrF | BLEU | chrF | BLEU |
| opus-mt (Reference) | 56.91 | 37.45 | — | — | — | — |
| Qwen + LoRA | 33.61 | 16.27 | **41.72** | **22.89** | *TBD* | *TBD* |
| Dream v2 (공식 SFT) | 19.51 | 9.79 | *TBD* | *TBD* | *TBD* | *TBD* |

> Qwen 1k→10k: chrF +8.1 (+24%), BLEU +6.6 (+41%). 데이터 10x 증가로 확실한 개선. 100k 실험으로 수확 체감 지점 확인 필요.

#### 10k test set 기준 (참고)

| 모델 | 데이터 | chrF | BLEU | Test set |
|------|:------:|:----:|:----:|:--------:|
| Qwen + LoRA | 10k | 36.00 | 12.80 | test_10k (1000문장) |

### 3-5. LoRA r=16 적정성 의견

현재 설정 `r=16, alpha=32`:
- 7B 모델 대비 trainable 비율 0.13% (10M params) — 번역 SFT에 적절한 범위
- Qwen 1k에서 chrF 33.6 달성 → r=16으로 충분히 학습 가능
- r=8이면 용량 부족으로 복잡 문법 학습 제한 가능, r=32~64는 과적합 위험 증가
- **결론**: r=16은 7B 모델의 번역 SFT에 적절. 데이터 10k 이상 시에도 r=16 유지 권장, 데이터 50k+ 시 r=32 검토.

### 3-6. 충분한 학습을 위한 데이터/학습 길이 의견

| 요인 | 현재 | 권장 | 근거 |
|------|------|------|------|
| 데이터 분량 | 10k | 20k~50k | Qwen 1k→10k에서 chrF +24% 확인. 50k 이상은 수확 체감 예상 |
| Epochs | 3 | 3~5 | 10k 이상이면 3 epoch 충분. 과적합 시 early stop |
| 예상 학습시간 (Qwen 10k) | 13.3분 | — | 50k 시 약 60~70분 예상 |
| 예상 학습시간 (Dream 10k) | ~30분 예상 | — | Custom loop이므로 Qwen 대비 2~3x |

**Dream이 Qwen보다 학습이 느린 이유**:
1. Custom training loop (TRL의 최적화 없음)
2. 4D attention mask 생성 오버헤드
3. q_sample() + time reweighting 계산
4. 매 step마다 랜덤 timestep 샘플링 및 마스킹

---

## 4. 다른 언어 SFT를 위한 데이터셋 가이드

### 4-1. 데이터 형식 (JSONL)

```json
{"instruction": "Translate to [TARGET_LANG]: [SOURCE_TEXT]", "output": "[TARGET_TEXT]", "en": "[SOURCE_TEXT]", "[lang_code]": "[TARGET_TEXT]"}
```

예시 (퀘냐):
```json
{"instruction": "Translate to Quenya: The king came to the city.", "output": "I aran túle i ostonna.", "en": "The king came to the city.", "qya": "I aran túle i ostonna."}
```

### 4-2. 권장 분량 및 분할

| 목적 | 최소 | 권장 | 분할 비율 |
|------|------|------|-----------|
| 파이프라인 검증 | 1k | 1k | 80/10/10 |
| 실질적 학습 | 5k | 10k~50k | 80/10/10 |
| 고품질 결과 | 20k | 50k+ | 90/5/5 |

### 4-3. 문장 길이

| 항목 | 권장 |
|------|------|
| 평균 길이 | 10~30 tokens (source 기준) |
| 최대 길이 | max_seq_length 이하 (현재 256) |
| 분포 | 다양해야 함. 짧은 문장만 있으면 긴 문장 생성 불가 |

### 4-4. 품질 요구사항

- 병렬 코퍼스 정합성: source-target 1:1 대응 필수
- 중복 제거: 동일 문장쌍 반복 시 과적합
- 도메인 다양성: 단일 도메인(예: 법률)만 있으면 범용성 저하
- 인공어의 경우: 문법 규칙 일관성 > 양. 불일치 데이터는 학습 방해

### 4-5. 실행 방법

```bash
# 데이터 준비 후
python3 train_qwen_10k.py   # Qwen (DATA_TAG 변수만 수정)
python3 train_dream_official.py --train_file data/train_[LANG].jsonl --val_file data/val_[LANG].jsonl
```

---

## 5. 정성 비교 (1k 데이터 기준)

| EN | 정답 (FI) | opus-mt | Qwen | Dream v2 |
|----|-----------|---------|------|----------|
| Yes! | Kyllä! | Jes! | Ja! | Joo! |
| I don't make the rules. | En laadi sääntöjä. | Minä en laadi sääntöjä. | En tehdä sääntöjä. | Minä en tekeä sääntöjä. |
| How we doing? | Miten sujuu? | Miten menee? | Miten meidän on? | Miten toimim? |
| Are you sure? - Yeah. | Oletko varma? | Oletko varma? | Oletko varma? | Olet varma? - Ja. |
| What are you going to do? | Eroatko sinä? | Mitä aiot tehdä? | Mitä teette? | Mitä tehdät? |

---

## 6. 결론

1. **공식 SFT 포팅으로 Dream 학습 정상화**: chrF 7→19.5 (1k 데이터). 학습 방식 정확성이 데이터 양보다 중요.
2. **AR vs Diffusion 구조적 차이**: 번역(순차 생성)은 AR 유리. Dream 강점은 planning 태스크.
3. **단일 환경 가능**: transformers==4.46.2에서 두 모델 모두 동작.
4. **10k 결과 TBD**: Dream 10k 학습 진행 중. 완료 후 본 보고서 업데이트.

---

*2026-05-09 | 10k 결과 업데이트 예정*
