# Xenolinguistics: Dream-7B vs Qwen2.5-7B LoRA Fine-Tuning

> Diffusion LLM(Dream-7B)과 Autoregressive LLM(Qwen2.5-7B)의 LoRA SFT 가능성을 EN→FI 번역 태스크로 비교 검증한 실험 보고서.

---

## 1. 실험 목적

핀란드어(Finnish)는 교착어로 15개 격변화를 가지며, 인공어(퀘냐, 클링온 등) 실험의 파이프라인 검증에 적합하다. 본 실험은 다음을 검증한다:

1. **Diffusion LLM에 LoRA SFT가 가능한가?**
2. **AR LLM 대비 번역 품질은 어떠한가?**
3. **단일 Python 환경에서 두 모델을 동시에 운영할 수 있는가?**

## 2. 실험 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX PRO 6000 Blackwell (98GB VRAM) |
| PyTorch | 2.10.0+cu128 |
| Transformers | 4.46.2 (Dream 호환 버전) |
| PEFT | 0.19.1 |
| TRL | 1.3.0 |
| 데이터셋 | Helsinki-NLP/opus-100 (en-fi) |

## 3. 모델 구성

| 역할 | 모델 | 방식 | 파라미터 |
|------|------|------|---------|
| **Reference NMT** | Helsinki-NLP/opus-mt-tc-big-en-fi | 전용 NMT, 학습 없음 | 237.7M |
| **AR LLM** | Qwen/Qwen2.5-7B-Instruct | LoRA r=16, causal LM SFT | 7.6B (trainable: 10M, 0.13%) |
| **Diffusion LLM** | Dream-org/Dream-v0-Instruct-7B | LoRA r=16, masked diffusion SFT | 7.6B (trainable: 10M, 0.13%) |

## 4. 학습 설정

### Qwen2.5-7B (AR)

| 항목 | 값 |
|------|-----|
| 데이터 | 1k (train 800 / val 100 / test 100) |
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-4 (cosine) |
| 학습 시간 | **0.8분** |
| Final train loss | 1.14 |
| Final eval loss | 1.43 |
| Format | Qwen ChatML (`<\|im_start\|>user/assistant`) |

### Dream-7B (Diffusion)

| 항목 | 값 |
|------|-----|
| 데이터 | 10k (train 8000 / val 1000 / test 100) |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-4 (cosine) |
| 학습 시간 | **7.3분** |
| Final train loss | 5.52 |
| Final val loss | 5.25 |
| Format | Dream ChatML (`apply_chat_template`) |
| Masking 전략 | **Response-only** (assistant 답변 토큰만 mask) |
| Mask ratio | Random 0.15~0.85 per batch |

### 학습 곡선

| Qwen Loss Curve | Dream Loss Curve |
|:---:|:---:|
| ![Qwen](outputs/qwen/loss_curve_1k.png) | ![Dream](outputs/dream/loss_curve_1k.png) |

## 5. 평가 결과

### 정량 비교

| 모델 | 모드 | chrF | BLEU | 데이터 | 비고 |
|------|------|:----:|:----:|:------:|------|
| **opus-mt (Reference NMT)** | — | **56.91** | **37.45** | — | 전용 NMT, fine-tuning 없음 |
| **Qwen2.5-7B + LoRA** | AR 생성 | **33.61** | **16.27** | 1k | 의미 있는 핀란드어 생성 |
| Dream-7B + LoRA | gt_length | 8.93 | 2.91 | 10k | GT 토큰 수 제공 |
| Dream-7B + LoRA | free_length | 7.03 | 0.83 | 10k | 길이 미지정 (64 토큰) |

### 정성 비교 (동일 입력)

| EN | 정답 (FI) | opus-mt | Qwen | Dream (gt_len) |
|----|-----------|---------|------|----------------|
| Yes! | Kyllä! | Jes! | Ja! | - O |
| I don't make the rules. | En laadi sääntöjä. | Minä en laadi sääntöjä. | En tehdä sääntöjä. | En teet. |
| How we doing? | Miten sujuu? | Miten menee? | Miten meidän on? | Miten meidän? |
| Thank you. | Kiitos. | Kiitos. | Kiitos. | Kiitos. |
| Are you sure? - Yeah. | Oletko varma? | Oletko varma? | Oletko varma? | Oletko varm |
| What are you going to do? | Eroatko sinä? | Mitä aiot tehdä? | Mitä teet? | Mitä teet? |

### 해석

- **Qwen (AR)**: 1k 데이터, 0.8분 학습으로 즉시 사용 가능한 수준의 핀란드어 생성. 문법 오류 있으나 의사소통 가능.
- **Dream (Diffusion)**: 10k 데이터, 7.3분 학습 후에도 일관된 문장 생성에 어려움. 단어 수준에서는 핀란드어 학습 증거 확인 ("Kiitos", "Oletko varm", "Mitä teet").
- **Reference NMT**: 전용 모델답게 압도적 성능.

## 6. 기술적 발견사항

### Dream-7B 호환성 이슈 (해결됨)

| 이슈 | 원인 | 해결 |
|------|------|------|
| `ROPE_INIT_FUNCTIONS['default']` KeyError | transformers 5.0에서 `'default'` 키 제거됨 | 4.46.2로 다운그레이드 |
| `diffusion_generate` 미사용 | 공식 API 대신 자체 denoising 구현 | 공식 `diffusion_generate()` + `DreamGenerationConfig` 사용 |
| 학습-추론 format 불일치 | 학습: plain text, 추론: chat_template | 학습도 `apply_chat_template` 사용 |
| 전체 토큰 masking | prompt도 mask → capacity 낭비 | **Response-only masking** (assistant 토큰만) |
| PeftModel GenerationConfig 충돌 | PeftModel이 standard GenerationConfig 사용 | `DreamGenerationConfig`를 명시적 전달 |
| Dream tokenizer `build_inputs_with_special_tokens` None | cls/sep 토큰이 None | `add_special_tokens=False` + 수동 padding |
| 생성 결과 전부 EOS | format 불일치로 모든 mask→EOS 디코딩 | chat_template 일치 + `<\|im_end\|>` 절단 |

### `diffusion_generate` 핵심 파라미터

```python
model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    generation_config=DreamGenerationConfig(
        max_new_tokens=64,
        steps=128,           # diffusion timesteps
        temperature=0.2,     # sampling temperature
        top_p=0.95,          # nucleus sampling
        alg="entropy",       # remasking: entropy > maskgit_plus > origin
        alg_temp=0.,         # confidence randomness
        mask_token_id=151666,
    ),
    return_dict_in_generate=True,
)
```

## 7. 결론

### LoRA Fine-Tuning 가능성

| 모델 | LoRA 적용 | SFT 수렴 | 번역 품질 |
|------|:---------:|:--------:|:---------:|
| Qwen2.5-7B | ✅ | ✅ | ✅ 실용 수준 |
| Dream-7B | ✅ | ✅ | ⚠️ 단어 수준만 |

### 핵심 결론

1. **Diffusion LLM의 LoRA SFT는 기술적으로 가능하다.** Loss는 정상 수렴한다.
2. **그러나 번역 품질에서 AR LLM에 크게 뒤진다.** Qwen은 800 샘플로 chrF 33을 달성한 반면, Dream은 8000 샘플로도 chrF 9에 불과하다.
3. **Dream의 병목**: (a) 출력 길이 예측 불가 (gt_length 제공 시 성능 향상), (b) bidirectional denoising에서 반복 패턴 생성, (c) AR 대비 ~10배 낮은 데이터 효율.
4. **단일 환경 운영**: `transformers==4.46.2` + `torch==2.10.0`에서 두 모델 모두 정상 동작 확인.

### 인공어 실험 확장성

핀란드어 파이프라인 검증 완료. 클링온/퀘냐/칼라니어 실험은 데이터 파일만 교체하면 동일 파이프라인 재사용 가능. 다만 **Dream은 현재 수준에서 인공어 번역에 비추천** — AR 모델(Qwen) 기반으로 진행하는 것이 효율적.

---

## 8. 파일 구조

```
Xenolinguistics/
├── README.md                          # 이 보고서
├── CLAUDE.md                          # 실험 지시서
├── data/                              # 데이터셋 (gitignored)
│   ├── train_1k.jsonl / val / test
│   └── train_10k.jsonl / val / test
├── outputs/
│   ├── reference/eval_results.json    # opus-mt 평가 결과
│   ├── qwen/
│   │   ├── adapter/                   # LoRA weights (gitignored)
│   │   ├── training_meta.json
│   │   ├── eval_results.json
│   │   └── loss_curve_1k.png
│   └── dream/
│       ├── adapter/                   # LoRA weights (gitignored)
│       ├── training_meta.json
│       ├── eval_results.json
│       └── loss_curve_1k.png
└── logs/
    ├── smoke_test_result.txt
    ├── env_info.txt
    └── model_check.json
```

---

*2026-04-28 | Auto-generated report*
