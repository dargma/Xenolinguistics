# Xenolinguistics

Diffusion LLM(Dream-7B)과 Autoregressive LLM(Qwen2.5-7B)의 LoRA SFT 비교 실험.
핀란드어(교착어, 15개 격변화)를 대상으로 번역 학습 파이프라인을 검증한다.

---

## 목적

1. Diffusion LLM에 LoRA SFT가 가능한가?
2. AR LLM 대비 번역 품질은 어떠한가?
3. 단일 Python 환경에서 두 모델을 동시에 운영할 수 있는가?

## 환경

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA RTX PRO 6000 Blackwell (98GB VRAM) |
| PyTorch | 2.10.0+cu128 |
| Transformers | 4.46.2 (Dream 호환 필수) |
| PEFT | 0.19.1 |
| TRL | 1.4.0 |

> Dream-7B는 transformers 5.x에서 `ROPE_INIT_FUNCTIONS['default']` KeyError 발생. 4.46.2 + HF 캐시 삭제로 해결.

## 모델

| 역할 | 모델 | 방식 |
|------|------|------|
| Reference NMT | Helsinki-NLP/opus-mt-tc-big-en-fi | 전용 NMT (학습 없음) |
| AR LLM | Qwen/Qwen2.5-7B-Instruct | LoRA r=16, causal LM SFT |
| Diffusion LLM | Dream-org/Dream-v0-Instruct-7B | LoRA r=16, 공식 SFT 포팅 |

> Dream-7B의 base LLM은 Qwen2.5-7B이다. 동일 base에서 생성 방식(AR vs Diffusion)만 다른 비교.

## 현재 결과 요약

| 모델 | 데이터 | chrF | BLEU |
|------|:------:|:----:|:----:|
| opus-mt (Reference) | — | 56.91 | 37.45 |
| Qwen + LoRA | 1k | 33.61 | 16.27 |
| Qwen + LoRA | 10k | 41.72 | 22.89 |
| Qwen + LoRA | 100k | 15.02 | 0.87 |
| Dream v2 + LoRA | 1k | 19.51 | 9.79 |
| Dream v2 + LoRA | 10k | 18.56 | 9.58 |

## 실험 보고서

- [2026-05-09: Qwen vs Dream 번역 능력 비교 (핀란드어)](reports/2026-05-09-dream-official-sft.md)

## 파일 구조

```
Xenolinguistics/
├── README.md                       # 프로젝트 개요
├── CLAUDE.md                       # 실험 지시서
├── train_dream_official.py         # Dream 공식 SFT 단일GPU 포팅
├── train_qwen_10k.py              # Qwen 10k 학습
├── eval_dream_official.py          # Dream 평가
├── plot_comparison.py              # Loss curve 비교 플롯
├── reports/                        # 실험 보고서
├── data/                           # 데이터셋 (gitignored)
├── outputs/                        # 학습 결과물
└── logs/                           # 환경 정보, 로그
```

## 참조

| 리소스 | URL |
|--------|-----|
| Dream-7B | [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) |
| Dream 공식 레포 | [DreamLM/Dream](https://github.com/DreamLM/Dream) |
| Qwen2.5-7B | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| 데이터셋 | [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) (en-fi) |
| Reference NMT | [Helsinki-NLP/opus-mt-tc-big-en-fi](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-fi) |

---

*2026-05-09*
