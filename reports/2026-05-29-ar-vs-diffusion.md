# 2026-05-29 — 인공어 AR vs Diffusion: 2 데이터셋 × 2 방향 종합

## 1. 요약

같은 백본 크기(7B)의 **AR LLM**(Qwen2.5-7B-Instruct)과 **Diffusion LLM**(Fast-dLLM v2 7B)이
인공어를 얼마나 잘 배우는지를, **두 인공어**(Klingon·Khalani) × **두 방향**(순방향 en→인공어 *생성*,
역방향 인공어→en *이해*)에서 **완전히 동일한 full-FT 설정**으로 비교했다. 평가는 두 모델에 같은 생성
예산을 주는 대칭 방식(free)과 정답 길이로 자르는 오라클(gt_length) 두 모드로 측정했다.

**핵심 발견 — 승자는 방향에 따라 뒤집힌다.**
- **순방향(en→tlh, 인공어를 만들어내기)**: AR이 유창성에서 우세하다(chrF 38.4 vs 35.7, BLEU 10.4 vs 2.6).
  반면 *어순* τ는 Diffusion이 더 높다(0.936 vs 0.909) — 맞은 단어를 정답 순서대로 놓는 능력은 양방향
  디코딩이 미세 우위. **혼합 결과.**
- **역방향(tlh→en, 인공어를 이해해 영어로 옮기기)**: **Diffusion이 chrF에서 AR을 역전한다(41.4 vs 37.2)**.
  출력이 모델이 이미 잘 아는 영어라, 양방향 모델이 영어 prior를 살려 더 유창한 문장을 만든다. 흥미롭게도
  이건 역방향 Diffusion의 **train loss가 훨씬 높게 끝났음에도(10.6 vs 순방향 3.8)** 나타난 현상.
- **종합**: "AR이 인공어에 본질적으로 낫다/Diffusion이 낫다"는 단정은 불가. **아키텍처 우위는 방향·지표에
  의존**한다. 어순(생성)에선 Diffusion, 영어 유창성(이해)에서도 Diffusion, 인공어 n-gram 정밀도(생성)에선 AR.
- **Khalani(55쌍)**: 양쪽 다 학습셋을 100% 암기(train loss ~0.17)하고 held-out에선 붕괴(chrF 13)
  → **교과서적 과적합**. 규모 한계로 AR/Diffusion 우열 판단 불가, "소량 인공어 FT = 암기" 사례로만 기록.

| 항목 | 값 |
|---|---|
| 날짜 | 2026-05-29 |
| 비교 축 | **2 데이터셋**(Klingon, Khalani) × **2 방향**(순 en→X, 역 X→en) × **2 아키텍처**(AR, Diffusion) |
| AR 모델 | `Qwen/Qwen2.5-7B-Instruct` |
| Diffusion 모델 | `Efficient-Large-Model/Fast_dLLM_v2_7B` |
| 환경 | `transformers 4.57.6`, `trl 1.4.0`, `torchao 0.17.0`, `peft 0.19.1` (5.x는 Fast-dLLM 생성이 깨짐) |

> 2026-05-28 보고서(`reports/2026-05-28-klingon-ar-vs-diffusion.md`)를 **다축 구조로 재구성·확장**한 supersede 판.
> 이전 보고서는 Klingon 순방향 + Khalani 탐색만 다뤘다. 본 보고서는 **역방향(X→en)** 축을 추가하고
> 전 실험을 단일 매트릭스로 통합한다.

### 1.1 마스터 결과 매트릭스 (free 모드, n=300)

같은 방향끼리만 비교. 숫자 = `outputs/*_ft_eval_*_free.json`. (gt_length·전체 수치는 §4.)

| 데이터셋 · 방향 | 메트릭 | AR (Qwen) | Diffusion (Fast-dLLM) | 우세 |
|---|---|:---:|:---:|:---:|
| **Klingon en→tlh** (순) | chrF | **38.43** | 35.67 | AR |
| | BLEU | **10.40** | 2.55 | AR |
| | 어순 τ | 0.909 | **0.936** | **Diffusion** |
| **Klingon tlh→en** (역) | chrF | 37.19 | **41.40** | **Diffusion** |
| | BLEU | **20.40** | 17.91 | AR |
| | 어순 τ | **0.934** | 0.922 | AR |
| **Khalani en→kha** (순, test n=11) | chrF | 13.20 | **14.94** | (노이즈) |
| | BLEU | **7.68** | 0.00 | (노이즈) |
| **Khalani kha→en** (역) | — | 🟡 학습중→TBD | 🟡 학습중→TBD | — |

### 1.2 실험 매트릭스 (8 런)

| # | 데이터셋 | 방향 | 아키텍처 | 런 디렉터리 | 학습 | 평가 |
|---|---|---|---|---|:---:|:---:|
| 1 | Klingon (12.7k) | en→tlh | AR | `klingon_qwen_fullft` | ✅ | ✅ |
| 2 | Klingon (12.7k) | en→tlh | Diffusion | `klingon_fastdllm_fullft` | ✅ | ✅ |
| 3 | Klingon (12.7k) | tlh→en | AR | `klingon_qwen_tlh2en_fullft` | ✅ | ✅ |
| 4 | Klingon (12.7k) | tlh→en | Diffusion | `klingon_fastdllm_tlh2en_fullft` | ✅ | ✅ |
| 5 | Khalani (55) | en→kha | AR | `khalani_qwen_fullft` | ✅ | ✅ |
| 6 | Khalani (55) | en→kha | Diffusion | `khalani_fastdllm_fullft` | ✅ | ✅ |
| 7 | Khalani (55) | kha→en | AR | `khalani_qwen_kha2en_fullft` | 🟡 | ⏳ TBD |
| 8 | Khalani (55) | kha→en | Diffusion | `khalani_fastdllm_kha2en_fullft` | 🟡 | ⏳ TBD |

HF 체크포인트(공개): 순방향 [AR](https://huggingface.co/sungkwang2/klingon-en2tlh-qwen2.5-7b-fullft)·[Diffusion](https://huggingface.co/sungkwang2/klingon-en2tlh-fastdllm-v2-7b-fullft) ·
역방향 [AR](https://huggingface.co/sungkwang2/klingon-tlh2en-qwen2.5-7b-fullft)·[Diffusion](https://huggingface.co/sungkwang2/klingon-tlh2en-fastdllm-v2-7b-fullft).

---

## 2. 데이터

| 데이터셋 | 출처 / 라이선스 | 규모 (train/val/test) | 어순 | 분할 스크립트 |
|---|---|---|---|---|
| **Klingon** (tlh) | OPUS **Tatoeba en-tlh** v2023-04-12 / CC-BY | 12,717 / 500 / 500 (중복제거 13,717쌍) | **OVS** (목적어-동사-주어) | `prepare_klingon.py` |
| **Khalani** (kha) | 자체 수집·번역 (무명 인공어, 게임/창작 코퍼스) | 55쌍 → 5-fold CV (fold0: 44 / 11) | (소량이라 어순 분석 불가) | `prepare_khalani.py` |

**왜 이 두 언어인가** — "미지 언어" 전제를 **두 극단**으로 잡았다:
- **Klingon**: 실제 사용자·코퍼스가 있는 인공어. 문장 수가 충분(12.7k)해 FT 결론이 가능하지만,
  온라인 자료가 많아 Qwen 사전학습에 **일부 포함됐을 가능성**(전제 오염, §7 caveat 1).
- **Khalani**: 사실상 무명이라 사전학습 오염이 거의 없어 "미지 언어" 전제엔 가장 깨끗하지만,
  **55쌍**뿐이라 FT로는 암기/과적합만 관측됨(§4.2). → 두 축은 **상보적**.

**스키마** (각 줄, 양 방향 공통):
```json
{"instruction": "Translate to Klingon: <EN>", "output": "<TLH>", "en": "<EN>", "klingon": "<TLH>"}
```
역방향 데이터는 순방향 jsonl의 instruction/output을 스왑(`*_tlh2en.jsonl`, instruction =
`"Translate to English: <tlh>"`). 동일 문장쌍이라 순↔역은 **완전 대칭**.

**토큰 길이 분포** (Qwen2.5 토크나이저, §3.2 절단 검증의 근거): 출력 토큰 max — Klingon FWD 42 /
REV 29 / Khalani 10 (test 기준). 전체(프롬프트+응답) max 224. 모두 짧아 학습 cap 512·평가 cap 64에
한참 못 미침.

---

## 3. 방법

### 3.1 학습 (전 런 동일)

| | 값 |
|---|---|
| 방식 | **full fine-tune** (`lora_rank=0`) |
| LR | 2e-5 |
| Epochs | 1 (Khalani만 데이터 보정 위해 20) |
| Eff. batch | 8 (AR: bs 8×accum 1 / Diffusion: bs 1×accum 8) |
| max_len | 512 |

LR·epoch·effective batch를 **동일**하게 맞춰 아키텍처만 변수로 분리. LoRA는 배제 — 선행 en→fi
실험에서 Fast-dLLM은 LoRA로 거의 학습되지 않아(rank를 키워도) Diffusion에 불리한 핸디캡이 되기 때문.
full-FT가 유일한 공정 공통 기반.

### 3.2 평가 — 생성/길이 처리 (AR vs Diffusion)

스크립트 `eval_ft.py`, **0-shot**(instruction만). 핵심 공정성 원칙은 선행 en→fi 실험에서 합의한 것과
동일: **두 아키텍처에 동일한 생성 예산(max_new_tokens=64)을 줘서 Diffusion의 denoise-budget 비대칭을
제거**한다. test 출력 최대가 42토큰이라 64는 충분(절단 0%, 학습 cap 512도 최대 224라 절단 0%).

| | **AR (Qwen)** | **Diffusion (Fast-dLLM v2)** |
|---|---|---|
| 디코딩 | autoregressive greedy (`do_sample=False`) | block diffusion `mdm_sample` (block 32, threshold 0.9) |
| 종료 (자동 길이) | EOS에서 **per-token 정지** | EOS(`stop_token=151645`) 감지 → 뒤 pad + 남은 블록 조기종료. **per-block(32)** granularity |
| 후처리 | 첫 비어있지 않은 줄 | 동일 |

**두 모드** (`--mode`): 동일 64 예산으로 생성 후 후처리만 다름.
- `free` (주력): 모델 자율 길이. 양쪽 다 정답 길이 모름 → 가장 공정한 자연길이 비교.
- `gt_length` (오라클): 생성 토큰을 **정답 토큰 길이로 슬라이스**(`gen_ids[:gt_tok]`, en→fi `eval_qwen.py`와
  동일 로직). 예산(64)≥정답(≤42)이라 "예산이 먼저 걸려 잘리는" 게 아니라 gt_length가 자른다.
  CLAUDE.md 하드룰 #2(학습 모델 free·gt_length 둘 다) 충족.

**Diffusion 생성의 step 수(denoising iteration)** — AR은 토큰당 1 forward로 고정이지만 Diffusion은 **동적**:
블록(32토큰)마다 매 step에서 confidence>0.9인 마스크 토큰을 한꺼번에 확정, 없으면 argmax로 최소 1개
(`fastdllm_generation.py` L96–128). 쉬운 블록은 적은 step, 어려운 블록은 많은 step. **실측(n=20/문장)**:

| 방향 | 평균 step/문장 | (min–max, median) | 평균 생성 길이 |
|---|:---:|:---:|:---:|
| 순방향 en→tlh | **30.1** | 23–41, med 29 | 35.8 tok |
| 역방향 tlh→en | **27.0** | 18–38, med 27 | 28.8 tok |

threshold 0.9에선 대략 토큰당 1 step 수준(생성 길이와 비슷). 캔버스 = `max_new_tokens 64 → 2블록`.

### 3.3 메트릭 정의

| 메트릭 | 정의 | 인공어에서의 의미 |
|---|---|---|
| **chrF** | 문자 n-gram F-score (`sacrebleu.corpus_chrf`, char 6-gram, β=2) | 접사·교착어 형태론에 강건 — Klingon 접사(`-pu'`,`vI-`,`-'a'`) 부분일치 포착. 주력. |
| **BLEU** | 단어 n-gram precision(≤4-gram)+brevity penalty | 정확 일치에 엄격. degenerate(반복) 출력에 가혹 → Diffusion 붕괴를 강하게 페널티. |
| **Exact match** | 후처리 후 예측==정답 비율(%) | 가장 엄격. 인공어에선 대체로 낮음. |
| **어순 τ** | 예측·정답 **둘 다 등장한 단어**의 순서에 대한 Kendall τ (≥2개 일치 쌍만) | **어휘와 분리된 순수 어순 신호**. 1.0=완벽, 0=무상관. |

---

## 4. 정량 평가

### 4.1 Klingon (12.7k) — free + gt_length, n=300

`outputs/klingon[_tlh2en]_ft_eval_{ar,diffusion}_{free,gt_length}.json`. **굵게 = 같은 모드·방향에서 우세.**

| 방향 | 모드 | 모델 | chrF | BLEU | EM | 어순 τ (n) |
|---|---|---|:---:|:---:|:---:|:---:|
| **en→tlh** (순) | free | AR | **38.43** | **10.40** | 8.33 | 0.909 (77) |
| | free | Diffusion | 35.67 | 2.55 | **8.67** | **0.936** (68) |
| | gt_length | AR | **36.96** | **10.15** | **8.67** | 0.904 (73) |
| | gt_length | Diffusion | 33.95 | 5.06 | 8.33 | **0.921** (55) |
| **tlh→en** (역) | free | AR | 37.19 | **20.40** | 7.33 | **0.934** (182) |
| | free | Diffusion | **41.40** | 17.91 | **9.00** | 0.922 (200) |
| | gt_length | AR | 36.34 | 19.51 | 7.33 | **0.924** (176) |
| | gt_length | Diffusion | **39.13** | **20.26** | **9.00** | 0.898 (189) |

읽는 법:
- **순방향**: chrF·BLEU는 AR, 어순 τ는 Diffusion (양 모드 일관). 인공어를 *만드는* 정밀도는 AR,
  맞은 단어의 *순서*는 Diffusion.
- **역방향**: chrF는 Diffusion이 역전(free 41.40>37.19). BLEU는 free에선 AR(20.40)이나 gt_length에선
  Diffusion(20.26)이 근소 우세 → 길이 정규화하면 Diffusion BLEU가 따라붙음. 어순 τ는 AR이 근소.
- 무학습 ICL 대비: 순방향 chrF가 ~10→38 (≈4배) → 12.7k FT로 인공어가 실제 학습됨.

### 4.2 Khalani (55쌍) — 과적합 실측 (탐색적)

fold0 = train 44 / test 11, full-FT, lr 2e-5, **epochs 20**. `outputs/khalani_ft_eval_*.json`.

| | AR train (44) | AR test (11) | Diffusion test (11) |
|---|:---:|:---:|:---:|
| chrF | **100.0** | 13.20 | 14.94 |
| BLEU | 100.0 | 7.68 | 0.00 |
| EM | **100%** | 0% | 0% |

학습셋 44개를 chrF/EM **100% 완전 암기**하지만 held-out에선 chrF 13으로 붕괴(격차 100→13) =
교과서적 과적합. held-out 차이(AR 13.2 vs DIFF 14.9)는 **n=11이라 노이즈**. → 55쌍은 결론 불가.

---

## 5. 정성 평가 (실제 예시)

### 5.1 Klingon en→tlh (순방향, `examples` 인용)

| English | Reference (tlh) | AR | Diffusion |
|---|---|---|---|
| I hate lawyers. | `chut qeSwI'pu' vImuS.` | `chut qeSwI'pu' vImuS.` ✅정확 | `chut qeSwI' vImuS.` |
| Are you coming to the store with me? | `ngevwI'Daq vIghoStaHvIS chotlhej'a'?` | `jIghoStaH'a' chaHvaD beylIjDaq?` | `ngevwIj vIjaHmeH chutlhe'taH'a'?` |
| What do we draw? | `nuq wIDIj?` | `nuq wIHIv?` | `nuq wIlo' 'e' wIvang?` |
| My mouth forms a lot of saliva. | `tlhepQe' law' lIng nujwIj.` | `tlhoS jIbIjqu' 'oH jatlhlu'wIj'e'.` | `, we have 1, 2, 3, 4, 5, …` ⚠️degenerate |

AR은 Klingon 형태론(접사 `-pu'`,`vI-`,`-'a'`)을 살림. Diffusion은 일부 입력에서 `, we have 1,2,3,…`
반복 붕괴 → BLEU를 크게 깎음(이것이 순방향 BLEU 2.55의 주원인).

### 5.2 Klingon tlh→en (역방향, `examples` 인용)

| Reference (en) | AR | Diffusion |
|---|---|---|
| They must be cops. | They will definitely be punished. | They were hungry enough to end the war. |
| This is the finest picture I have ever seen. | This picture is the most beautiful one I have ever seen | This picture is the best of all the pictures I've seen |
| Are you coming to the store with me? | May I go with you to the museum? | Would you come with me to the back of the store? |
| I like to look at old pictures. | I like looking at pictures of animals. | I like taking pictures of the stars. |

양쪽 다 **문법적으로 자연스러운 영어**를 생성하지만 의미는 자주 빗나감(인공어 입력 이해 한계).
"store with me" 예시처럼 Diffusion이 핵심어(store, come with me)를 더 자주 보존 → 역방향 chrF 우세와 일치.

### 5.3 Khalani en→kha
`Prismatic core online` → ref `Peradak kural`, AR `Peradak aghanizha` (첫 단어만 맞고 뒤는 암기 어휘).

---

## 6. Loss 곡선

4-패널 = 아키텍처(AR/Diffusion) × 데이터셋(Klingon/Khalani). Klingon 패널은 순·역을 겹쳐 그리고,
AR은 **eval loss(점선)** 도 함께. Diffusion 트레이너엔 val 셋이 없어 train만. 생성기: `reports/figures/2026-05-29-loss.py`.

![loss](figures/2026-05-29-loss.png)

| 런 | train loss (시작→끝) | eval loss 커브 |
|---|---|---|
| Klingon en→tlh AR | 6.05 → **1.044** | 1.677→1.444→1.285→1.193→1.146→1.133→**1.129** (단조↓, 과적합 없음) |
| Klingon en→tlh Diffusion | 12.67 → **3.81** | (val 없음) |
| Klingon tlh→en AR | → **1.148** | 1.442→1.256→**1.234** (단조↓) |
| Klingon tlh→en Diffusion | 28 → **~10.6** (구간평균 16.35) | (val 없음) |
| Khalani en→kha AR | → **0.17** (44개 암기) | (val 없음) |
| Khalani en→kha Diffusion | → **0.19** (44개 암기) | (val 없음) |

- **핵심**: Klingon Diffusion 패널에서 순방향(→3.81)은 잘 내려가지만 **역방향은 ~10.6에서 정체** —
  train loss만 보면 역방향 diffusion이 더 안 맞는 듯하나, 정작 **평가 chrF는 역방향 Diffusion이 더 높다**
  (§4.1). train loss(인공어 입력 denoising 난이도)와 출력 품질(영어 유창성)이 분리되는 흥미로운 지점.
- Khalani 양쪽 train loss ~0.17–0.19로 0 근처 추락 = 44개 완전 암기 → 과적합 시각화(§4.2).

---

## 7. 종합 해석 & 주의

**해석.** 신뢰할 만한 비교는 **Klingon 양방향** 축에서 나온다. 결론은 *방향 의존적*이다 — 순방향(생성)에선
어순은 Diffusion·유창성은 AR, 역방향(이해)에선 영어 chrF가 Diffusion으로 역전. "한 아키텍처가 인공어에
일관되게 우월"하다는 증거는 없다. 양방향 디코딩의 이점은 **어순 배치**와 **출력측 영어 유창성**에서
관찰되고, AR의 이점은 **인공어 n-gram 정밀도**에서 관찰된다. Khalani는 규모 한계로 암기만 확인.

**주의 (caveat).**
1. Klingon은 온라인 자료가 많아 **Qwen 사전학습 포함 가능성** → 부분적으로 *회상*(미지 언어 전제 오염).
   더 깨끗한 축은 Khalani(단 55쌍이라 FT 부적합).
2. single seed, Diffusion 샘플링 비결정적 → 작은 τ 격차 신뢰엔 ≥3 seed 필요.
3. 1 epoch, Diffusion 디코딩 하이퍼파라미터(block_size 32 / threshold 0.9) 미튜닝.
4. 어순 τ는 ≥2단어 일치 쌍에서만 계산(역방향은 영어 출력이라 매칭 단어가 많아 n이 큼 vs 순방향은 적음).
   역방향 τ 해석은 어휘 중복 효과 주의.

---

## 8. 재현

환경: `transformers 4.57.6`, `trl 1.4.0`, `torchao 0.17.0`, `peft 0.19.1`. 모델·데이터는
`/content/local_fast`에서만 읽음(Drive는 백업 전용).

```bash
# 데이터: 순방향 en→tlh
python3 prepare_klingon.py
# 역방향 tlh→en: 순방향 jsonl의 instruction/output 스왑 → data/klingon/*_tlh2en.jsonl

# 학습 (순방향 예시; 역방향은 train_file만 *_tlh2en.jsonl 로)
python3 train/train_qwen_v4.py   --train_file data/klingon/train.jsonl --val_file data/klingon/val.jsonl \
  --lora_rank 0 --lr 2e-5 --epochs 1 --output_dir outputs/klingon_qwen_fullft           # AR
python3 train/train_fastdllm.py  --train_file data/klingon/train.jsonl \
  --lora_rank 0 --lr 2e-5 --epochs 1 --batch_size 1 --grad_accum 8 --max_len 512 \
  --output_dir outputs/klingon_fastdllm_fullft                                          # Diffusion

# 평가 (방향별 --tgt_field/--lang_name/--test_file, 모드별 --mode free|gt_length)
PYTHONPATH=eval python3 eval_ft.py --model_type ar \
  --model_path outputs/klingon_qwen_fullft/final --n_eval 300 --mode free \
  --out outputs/klingon_ft_eval_ar_free.json
PYTHONPATH=eval python3 eval_ft.py --model_type diffusion \
  --model_path outputs/klingon_fastdllm_fullft/final --n_eval 300 --mode free \
  --out outputs/klingon_ft_eval_diffusion_free.json
```
