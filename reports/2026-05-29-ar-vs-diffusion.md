# 2026-05-29 — 인공어 AR vs Diffusion: 2 데이터셋 × 2 방향 종합

## 0. 목차

1. **요약** — 핵심 발견 · 메타데이터 · 결과/실험 매트릭스
2. **방법** — 2.1 데이터 · 2.2 학습(+Loss 곡선) · 2.3 평가(생성·길이·메트릭)
3. **결과** — 3.1 정량 · 3.2 정성
4. **논의** — 해석 · 주의(caveat)
5. **재현성** — 실행 스크립트
6. **링크** — HF 모델 · 데이터셋

---

## 1. 요약

같은 7B 백본의 **AR LLM**(Qwen2.5-7B-Instruct)과 **Diffusion LLM**(Fast-dLLM v2 7B)이 인공어를
얼마나 잘 배우는지를, **두 인공어**(Klingon 12.7k · Khalani 55쌍) × **두 방향**(순방향 en→인공어 *생성*,
역방향 인공어→en *이해*) × **두 아키텍처**에서 **완전히 동일한 full-FT 설정**으로 비교했다.

**핵심 발견 — 승자는 방향에 따라 뒤집힌다.**
- **순방향(en→tlh, 인공어 생성)**: 유창성은 **AR 우세**(chrF 38.4 vs 35.7, BLEU 10.4 vs 2.6),
  *어순*은 **Diffusion 우세**(τ 0.936 vs 0.909). 혼합.
- **역방향(tlh→en, 인공어 이해)**: **Diffusion이 chrF를 역전**(41.4 vs 37.2). 출력이 모델이 이미 아는
  영어라 양방향 모델이 영어 prior를 살려 더 유창. 역방향 Diffusion의 **train loss가 훨씬 높게 끝났음에도**
  (10.6 vs 순방향 3.8) 나타난 현상 — 입력 denoising 난이도와 출력 품질이 분리됨.
- **종합**: "한 아키텍처가 인공어에 일관 우월"하다는 증거 없음. **우위는 방향·지표에 의존.**
- **Khalani(55쌍)**: 양방향 모두 학습셋 암기(train loss ~0.18) + held-out 붕괴(chrF 11–15, EM 0)
  = 교과서적 과적합. 규모 한계로 우열 판단 불가 — "소량 인공어 FT = 암기" 사례.

| 항목 | 값 |
|---|---|
| 날짜 | 2026-05-29 |
| 비교 축 | 2 데이터셋(Klingon, Khalani) × 2 방향(순 en→X, 역 X→en) × 2 아키텍처(AR, Diffusion) |
| AR / Diffusion 모델 | `Qwen/Qwen2.5-7B-Instruct` / `Efficient-Large-Model/Fast_dLLM_v2_7B` |
| 환경 | `transformers 4.57.6`, `trl 1.4.0`, `torchao 0.17.0`, `peft 0.19.1` (5.x는 Fast-dLLM 생성이 깨짐) |

**마스터 결과 매트릭스** (free 모드, 같은 방향끼리만 비교; 전체 수치·gt_length는 §3.1):

| 데이터셋 · 방향 (n) | chrF (AR / Diff) | BLEU (AR / Diff) | 어순 τ (AR / Diff) |
|---|:---:|:---:|:---:|
| Klingon en→tlh (300) | **38.4** / 35.7 | **10.4** / 2.6 | 0.909 / **0.936** |
| Klingon tlh→en (300) | 37.2 / **41.4** | **20.4** / 17.9 | **0.934** / 0.922 |
| Khalani en→kha (11) | 13.2 / **14.9** | **7.7** / 0.0 | — |
| Khalani kha→en (11) | **11.8** / 11.1 | 2.4 / **2.7** | — |

**실험 매트릭스** (8 런, 전부 학습·평가 완료):

| 데이터셋 | 방향 | AR 런 | Diffusion 런 |
|---|---|---|---|
| Klingon 12.7k | en→tlh | `klingon_qwen_fullft` | `klingon_fastdllm_fullft` |
| Klingon 12.7k | tlh→en | `klingon_qwen_tlh2en_fullft` | `klingon_fastdllm_tlh2en_fullft` |
| Khalani 55 | en→kha | `khalani_qwen_fullft` | `khalani_fastdllm_fullft` |
| Khalani 55 | kha→en | `khalani_qwen_kha2en_fullft` | `khalani_fastdllm_kha2en_fullft` |

체크포인트·데이터셋 HF 링크 → **§6**.

---

## 2. 방법

### 2.1 데이터

| 데이터셋 | 출처 / 라이선스 | 규모 (train/val/test) | 어순 |
|---|---|---|---|
| **Klingon** (tlh) | OPUS **Tatoeba en-tlh** v2023-04-12 / CC-BY (중복제거 13,717쌍) | 12,717 / 500 / 500 | **OVS** |
| **Khalani** (kha) | 자체 수집·번역 (무명 인공어, 게임/창작) | 55쌍 → 5-fold CV (fold0: 44 / 11) | (소량, 분석 불가) |

"미지 언어" 전제를 **두 극단**으로 잡았다: **Klingon**은 코퍼스가 충분(12.7k)해 FT 결론이 가능하나
온라인 자료가 많아 Qwen 사전학습에 일부 포함됐을 수 있다(전제 오염, §4). **Khalani**는 무명이라
오염은 거의 없지만 55쌍뿐이라 암기만 관측된다(§3.1). → 상보적.

**스키마**(각 줄): `{"instruction","output","en","klingon"}`. 역방향은 instruction/output을 스왑
(`*_tlh2en.jsonl`, `"Translate to English: <tlh>"`) — 동일 문장쌍이라 순↔역 **완전 대칭**.
생성: `prepare_klingon.py` / `prepare_khalani.py`.

### 2.2 학습

전 런 **동일 설정**으로 아키텍처만 변수로 분리. LoRA는 배제 — 선행 en→fi에서 Fast-dLLM이 LoRA로는
거의 학습되지 않아 Diffusion에 불리한 핸디캡이 되기 때문(full-FT가 유일한 공정 공통 기반).

| 방식 | LR | Epochs | Eff. batch | max_len |
|---|---|---|---|---|
| full fine-tune (`lora_rank=0`) | 2e-5 | 1 (Khalani만 20) | 8 (AR bs8×acc1 / Diff bs1×acc8) | 512 |

**Loss 곡선** (4-패널 = 아키텍처 × 데이터셋; Klingon은 순·역 겹쳐 그림, AR은 eval loss 점선도 포함.
Diffusion 트레이너엔 val 셋이 없어 train만). 생성기 `reports/figures/2026-05-29-loss.py`.

![loss](figures/2026-05-29-loss.png)

| 런 | train loss (시작→끝) | eval loss |
|---|---|---|
| Klingon en→tlh AR | 6.05 → **1.04** | 1.68→**1.13** (단조↓, 과적합 없음) |
| Klingon en→tlh Diff | 12.67 → **3.81** | (val 없음) |
| Klingon tlh→en AR | → **1.15** | 1.44→**1.23** |
| Klingon tlh→en Diff | 28 → **~10.6** (평균 16.4) | (val 없음) |
| Khalani 양방향 AR | → **~0.18** (44개 암기) | (val 없음) |
| Khalani 양방향 Diff | → 0.19 (순) / 6.8 (역, 평균) | (val 없음) |

읽을 점: Klingon Diffusion 패널에서 순방향(→3.81)은 잘 내려가나 **역방향은 ~10.6에서 정체** — 그럼에도
평가 chrF는 역방향 Diffusion이 더 높다(§3.1). Khalani는 양쪽 다 train loss가 0 근처로 추락 = 암기.

### 2.3 평가 — 생성·길이 처리

`eval_ft.py`, **0-shot**. 공정성 원칙(선행 en→fi와 동일): **두 아키텍처에 동일 생성 예산
(max_new_tokens=64)** 을 줘 Diffusion의 denoise-budget 비대칭을 제거. 학습·평가 모두 **절단 0%**
(학습 전체 max 224<512, test 출력 max 42<64).

| | **AR (Qwen)** | **Diffusion (Fast-dLLM v2)** |
|---|---|---|
| 디코딩 | autoregressive greedy | block diffusion `mdm_sample` (block 32, threshold 0.9) |
| 길이 자동조절 | EOS에서 per-token 정지 | EOS(`151645`) 감지 → 뒤 pad + 남은 블록 조기종료 (per-block) |
| 후처리 | 첫 비어있지 않은 줄 | 동일 |

**두 모드**(`--mode`): 동일 64 예산 생성 후 후처리만 — `free`(주력, 모델 자율 길이) / `gt_length`(오라클,
생성 토큰을 정답 길이로 슬라이스 `gen_ids[:gt_tok]`). 예산(64)≥정답(≤42)이라 gt_length가 실제 컷 기준.

**Diffusion step 수**: AR은 토큰당 1 forward로 고정, Diffusion은 동적(블록당 confidence>0.9 토큰을
한꺼번에 확정). **실측(n=20)**: 순방향 평균 **30.1** step/문장(23–41), 역방향 **27.0**(18–38) —
threshold 0.9에서 대략 토큰당 1 step. 캔버스 = 64토큰 = 2블록.

**메트릭**: **chrF**(문자 n-gram F, 교착어 형태론에 강건; 주력) · **BLEU**(단어 n-gram, degenerate에
가혹) · **EM**(완전일치 %) · **어순 τ**(예측·정답 공통 단어의 Kendall 순서상관, 어휘와 분리된 어순 신호;
≥2 공통단어 쌍만).

---

## 3. 결과

### 3.1 정량

**Klingon** (n=300). `outputs/klingon[_tlh2en]_ft_eval_{ar,diffusion}_{free,gt_length}.json`.
**굵게 = 같은 방향·모드에서 우세.**

| 방향 | 모드 | chrF (AR/Diff) | BLEU (AR/Diff) | EM (AR/Diff) | 어순 τ (AR/Diff) |
|---|---|:---:|:---:|:---:|:---:|
| en→tlh (순) | free | **38.43**/35.67 | **10.40**/2.55 | 8.33/**8.67** | 0.909/**0.936** |
| | gt_length | **36.96**/33.95 | **10.15**/5.06 | **8.67**/8.33 | 0.904/**0.921** |
| tlh→en (역) | free | 37.19/**41.40** | **20.40**/17.91 | 7.33/**9.00** | **0.934**/0.922 |
| | gt_length | 36.34/**39.13** | 19.51/**20.26** | 7.33/**9.00** | **0.924**/0.898 |

- 순방향: chrF·BLEU는 AR, 어순 τ는 Diffusion (양 모드 일관).
- 역방향: chrF는 Diffusion 역전. BLEU는 free에선 AR, gt_length에선 Diffusion(20.26)이 따라붙음.
- ICL 대비 순방향 chrF ~10→38(≈4배) → 12.7k FT로 인공어가 실제 학습됨.

**Khalani** (fold0, train 44 / test 11, epochs 20). `outputs/khalani[_kha2en]_ft_eval_*.json`.

| 방향 | chrF (AR/Diff) | BLEU (AR/Diff) | EM | (참고) AR train |
|---|:---:|:---:|:---:|:---:|
| en→kha | 13.20/**14.94** | **7.68**/0.00 | 0/0 | chrF/EM **100** |
| kha→en | **11.82**/11.08 | 2.41/**2.71** | 0/0 | — |

학습셋 44개를 100% 암기하나 held-out에선 chrF 11–15·EM 0으로 붕괴 → **교과서적 과적합**.
방향·아키텍처 차이는 **n=11이라 노이즈** — 55쌍은 결론 불가.

### 3.2 정성

**Klingon en→tlh** (순):

| English | Reference | AR | Diffusion |
|---|---|---|---|
| I hate lawyers. | `chut qeSwI'pu' vImuS.` | `chut qeSwI'pu' vImuS.` ✅ | `chut qeSwI' vImuS.` |
| What do we draw? | `nuq wIDIj?` | `nuq wIHIv?` | `nuq wIlo' 'e' wIvang?` |
| My mouth forms a lot of saliva. | `tlhepQe' law' lIng nujwIj.` | `tlhoS jIbIjqu'...` | `, we have 1, 2, 3, …` ⚠️degenerate |

AR은 형태론(접사 `-pu'`,`vI-`,`-'a'`)을 살림. Diffusion은 일부 입력에서 `, we have 1,2,3,…` 반복
붕괴 → 순방향 BLEU 2.55의 주원인.

**Klingon tlh→en** (역):

| Reference (en) | AR | Diffusion |
|---|---|---|
| Are you coming to the store with me? | May I go with you to the museum? | Would you come with me to the back of the store? |
| This is the finest picture I have ever seen. | This picture is the most beautiful one I have ever seen | This picture is the best of all the pictures I've seen |

양쪽 다 **문법적으로 자연스러운 영어**를 만들지만 의미는 자주 빗나감. Diffusion이 핵심어(store, come
with me)를 더 자주 보존 → 역방향 chrF 우세와 일치.

**Khalani**: `Prismatic core online` → ref `Peradak kural`, AR `Peradak aghanizha`(첫 단어만);
역방향 Diffusion은 `Prismaticismatic beams`처럼 반복 붕괴.

---

## 4. 논의

신뢰할 만한 비교는 **Klingon 양방향**에서 나오며, 결론은 *방향 의존적*이다 — 순방향(생성)은 어순=Diffusion·
유창성=AR, 역방향(이해)은 영어 chrF가 Diffusion으로 역전. 양방향 디코딩의 이점은 **어순 배치**와
**출력측 영어 유창성**에서, AR의 이점은 **인공어 n-gram 정밀도**에서 관찰된다. Khalani는 규모 한계로
암기만 확인. 흥미로운 분리: 역방향 Diffusion은 train loss가 높아도(10.6) 출력 영어 품질은 우수.

**주의 (caveat).**
1. Klingon은 온라인 자료가 많아 Qwen 사전학습 포함 가능성 → 부분적 *회상*(미지 언어 전제 오염). 더
   깨끗한 축은 Khalani지만 55쌍이라 FT 부적합.
2. single seed, Diffusion 샘플링 비결정적 → 작은 τ 격차 신뢰엔 ≥3 seed 필요.
3. 1 epoch, Diffusion 디코딩 하이퍼파라미터(block 32 / threshold 0.9) 미튜닝.
4. 어순 τ는 ≥2 공통단어 쌍에서만 계산. 역방향(영어 출력)은 어휘 중복이 커 n이 큼(순방향은 적음) →
   방향 간 τ 직접 비교는 주의.

---

## 5. 재현성

환경: `transformers 4.57.6` · `trl 1.4.0` · `torchao 0.17.0` · `peft 0.19.1`. 모델·데이터는
`/content/local_fast`에서만 읽음(Drive는 백업).

```bash
# 1) 데이터 (역방향은 instruction/output 스왑 → *_tlh2en.jsonl)
python3 prepare_klingon.py

# 2) 학습 — 순방향 예시 (역방향은 train_file만 *_tlh2en.jsonl)
python3 train/train_qwen_v4.py  --train_file data/klingon/train.jsonl --val_file data/klingon/val.jsonl \
  --lora_rank 0 --lr 2e-5 --epochs 1 --output_dir outputs/klingon_qwen_fullft           # AR
python3 train/train_fastdllm.py --train_file data/klingon/train.jsonl \
  --lora_rank 0 --lr 2e-5 --epochs 1 --batch_size 1 --grad_accum 8 --max_len 512 \
  --output_dir outputs/klingon_fastdllm_fullft                                          # Diffusion

# 3) 평가 — 방향별 --tgt_field/--lang_name/--test_file, 모드별 --mode free|gt_length
PYTHONPATH=eval python3 eval_ft.py --model_type ar        --model_path outputs/klingon_qwen_fullft/final \
  --n_eval 300 --mode free --out outputs/klingon_ft_eval_ar_free.json
PYTHONPATH=eval python3 eval_ft.py --model_type diffusion --model_path outputs/klingon_fastdllm_fullft/final \
  --n_eval 300 --mode free --out outputs/klingon_ft_eval_diffusion_free.json
```

---

## 6. 링크

**HF 모델 체크포인트** (공개, full-FT):

| 방향 | AR (Qwen2.5-7B) | Diffusion (Fast-dLLM v2 7B) |
|---|---|---|
| en→tlh | [klingon-en2tlh-qwen2.5-7b-fullft](https://huggingface.co/sungkwang2/klingon-en2tlh-qwen2.5-7b-fullft) | [klingon-en2tlh-fastdllm-v2-7b-fullft](https://huggingface.co/sungkwang2/klingon-en2tlh-fastdllm-v2-7b-fullft) |
| tlh→en | [klingon-tlh2en-qwen2.5-7b-fullft](https://huggingface.co/sungkwang2/klingon-tlh2en-qwen2.5-7b-fullft) | [klingon-tlh2en-fastdllm-v2-7b-fullft](https://huggingface.co/sungkwang2/klingon-tlh2en-fastdllm-v2-7b-fullft) |

**HF 데이터셋**: [sungkwang2/klingon-en-tlh-translation](https://huggingface.co/datasets/sungkwang2/klingon-en-tlh-translation) (양방향 jsonl, Tatoeba 기반 CC-BY).

> Khalani 모델(탐색적·과적합)은 로컬 보관, HF 미업로드. Khalani 데이터는 자체 코퍼스라 비공개.
