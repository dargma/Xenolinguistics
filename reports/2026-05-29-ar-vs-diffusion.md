# 2026-05-29 — 인공어 AR vs Diffusion: 2 데이터셋 × 2 방향 종합

| 항목 | 값 |
|---|---|
| 작성 | dargma |
| 날짜 | 2026-05-29 |
| 비교 축 | **2 데이터셋**(Klingon, Khalani) × **2 방향**(순방향 en→X, 역방향 X→en) × **2 아키텍처**(AR, Diffusion) |
| AR 모델 | `Qwen/Qwen2.5-7B-Instruct` |
| Diffusion 모델 | `Efficient-Large-Model/Fast_dLLM_v2_7B` |
| 환경 | `transformers 4.57.6`, `trl 1.4.0`, `torchao 0.17.0`, `peft 0.19.1` (5.x는 Fast-dLLM 생성이 깨짐) |

> 2026-05-28 보고서(`reports/2026-05-28-klingon-ar-vs-diffusion.md`)를 **다축 구조로 재구성·확장**한 supersede 판.
> 이전 보고서는 Klingon 순방향 + Khalani 탐색만 다뤘다. 본 보고서는 **역방향(X→en)** 축을 추가하고
> 전 실험을 단일 매트릭스로 통합한다.

---

## 1. 실험 매트릭스 (8 런 = 2 데이터셋 × 2 방향 × 2 아키텍처)

| # | 데이터셋 | 방향 | 아키텍처 | 런 디렉터리 | 학습 | 평가 |
|---|---|---|---|---|:---:|:---:|
| 1 | Klingon (12.7k) | en→tlh (순) | AR | `outputs/klingon_qwen_fullft` | ✅ | ✅ |
| 2 | Klingon (12.7k) | en→tlh (순) | Diffusion | `outputs/klingon_fastdllm_fullft` | ✅ | ✅ |
| 3 | Klingon (12.7k) | tlh→en (역) | AR | `outputs/klingon_qwen_tlh2en_fullft` | ✅ | ⏳ TBD |
| 4 | Klingon (12.7k) | tlh→en (역) | Diffusion | `outputs/klingon_fastdllm_tlh2en_fullft` | 🟡 학습중 | ⏳ TBD |
| 5 | Khalani (55) | en→kha (순) | AR | `outputs/khalani_qwen_fullft` | ✅ | ✅ |
| 6 | Khalani (55) | en→kha (순) | Diffusion | `outputs/khalani_fastdllm_fullft` | ✅ | ✅ |
| 7 | Khalani (55) | kha→en (역) | AR | — | ⬜ TBD | ⬜ TBD |
| 8 | Khalani (55) | kha→en (역) | Diffusion | — | ⬜ TBD | ⬜ TBD |

범례: ✅완료 · 🟡진행중 · ⏳학습완료-평가대기 · ⬜미착수.

### 1.1 마스터 결과 매트릭스 (한눈에)

같은 방향끼리만 비교(`free`-length 대칭). 숫자 = `outputs/*_ft_eval_*.json` (n은 셀별 표기).

| 데이터셋 · 방향 | 메트릭 | AR (Qwen) | Diffusion (Fast-dLLM) | H1 관련 우세 |
|---|---|:---:|:---:|:---:|
| **Klingon en→tlh** (n=300) | chrF | **38.43** | 35.67 | AR(근소) |
| | BLEU | **10.40** | 2.55 | AR |
| | EM | 8.33 | **8.67** | ≈ |
| | **어순 τ** | 0.909 | **0.936** | **Diffusion** |
| **Klingon tlh→en** (n=TBD) | chrF | ⏳ TBD | ⏳ TBD | TBD |
| | BLEU | ⏳ TBD | ⏳ TBD | TBD |
| | EM | ⏳ TBD | ⏳ TBD | TBD |
| | 어순 τ | ⏳ TBD | ⏳ TBD | TBD |
| **Khalani en→kha** (test n=11) | chrF | 13.20 | **14.94** | (노이즈, n=11) |
| | BLEU | **7.68** | 0.00 | (노이즈) |
| | EM | 0.0 | 0.0 | = |
| **Khalani kha→en** | — | ⬜ TBD | ⬜ TBD | TBD |

---

## 2. 가설 (H1)

인공어는 어휘뿐 아니라 **어순**도 영어와 다르고 LLM 사전학습에 (대체로) 포함되지 않는다.
양방향 denoising으로 디코딩하는 **Diffusion LLM**이 좌→우 단방향 **AR LLM**보다 인공어를 —
특히 어순을 — 더 잘 학습할 것이다.

**방향 축의 의미**: 순방향(en→인공어)은 *인공어 생성* — 어순·형태론을 모델이 직접 만들어야 하므로
H1이 가장 직접적으로 검증되는 방향이다. 역방향(인공어→en)은 *인공어 이해* — 출력은 모델이 이미 잘 아는
영어라, 양방향 인코딩이 **입력 이해**에 주는 이점만 분리해서 본다. 두 방향을 같이 보면 H1의 효과가
"생성"에서 오는지 "이해"에서 오는지 구분할 수 있다.

---

## 3. 공통 설정

### 3.1 데이터

| 데이터셋 | 출처 / 라이선스 | 규모 (train/val/test) | 어순 | 분할 스크립트 |
|---|---|---|---|---|
| **Klingon** (tlh) | OPUS **Tatoeba en-tlh** v2023-04-12 / CC-BY | 12,717 / 500 / 500 (중복제거 13,717쌍) | **OVS** (목적어-동사-주어) | `prepare_klingon.py` |
| **Khalani** (kha) | 자체 수집·번역 (무명 인공어, 게임/창작 코퍼스) | 55쌍 → 5-fold CV (fold0: 44 / 11) | (소량이라 어순 분석 불가) | `prepare_khalani.py` |

**왜 이 두 언어인가** — H1의 "미지 언어" 전제를 **두 극단**으로 잡았다:
- **Klingon**: 실제 사용자·코퍼스가 있는 인공어. 문장 수가 충분(12.7k)해 FT 결론이 가능하지만,
  온라인 자료가 많아 Qwen 사전학습에 **일부 포함됐을 가능성**(전제 오염, §9 caveat 1).
- **Khalani**: 사실상 무명이라 사전학습 오염이 거의 없어 "미지 언어" 전제엔 가장 깨끗하지만,
  **55쌍**뿐이라 FT로는 암기/과적합만 관측됨(§4.3). → 두 축은 **상보적**.

**스키마** (각 줄, 양 방향 공통):
```json
{"instruction": "Translate to Klingon: <EN>", "output": "<TLH>", "en": "<EN>", "klingon": "<TLH>"}
```
- **역방향 데이터**: 순방향 jsonl의 instruction/output을 스왑 (`data/klingon/{train,val,test}_tlh2en.jsonl`,
  instruction = `"Translate to English: <tlh>"`, output = `<en>`). 동일 문장쌍이라 순↔역은 **완전 대칭**.

**토큰 길이 분포** (Qwen2.5 토크나이저 기준, §3.5 절단 검증의 근거):

| 데이터 | 출력 토큰 max / p50 | 전체(프롬프트+응답) max |
|---|---|---|
| Klingon FWD (out=tlh) | 42 / 11 (test), 115 / 11 (train) | 224 |
| Klingon REV (out=en) | 29 / 6 (test), 73 / 6 (train) | 223 |
| Khalani (out=kha) | 10 / 5 | — |

- Khalani 구성: Phrases 34 · Single Words/Terms 20 · Affirmations 1 (예: `Oblivion awaits` → `Zerashk Gulida`).
- 모든 출력이 짧음(test 기준 ≤42 tok) → 학습 cap 512·평가 cap 64에 한참 못 미침(§3.5).

### 3.2 학습 (전 런 동일)
| | 값 |
|---|---|
| 방식 | **full fine-tune** (`lora_rank=0`) |
| LR | 2e-5 |
| Epochs | 1 (Khalani만 데이터 보정 위해 20) |
| Eff. batch | 8 (AR: bs 8×accum 1 / Diffusion: bs 1×accum 8) |
| max_len | 512 |

LR·epoch·effective batch를 **동일**하게 맞춰 아키텍처만 변수로 분리. LoRA는 배제 — 선행 en→fi
실험에서 Fast-dLLM은 LoRA로 거의 학습되지 않아(rank를 키워도) LoRA 비교는 Diffusion에 불리한
핸디캡이 되기 때문. full-FT가 유일한 공정 공통 기반.

### 3.3 평가 — 생성/길이 처리 (AR vs Diffusion)

스크립트 `eval_ft.py`, **0-shot**(instruction만), **양쪽 free-length 대칭**. 핵심 공정성 원칙은
선행 en→fi(핀란드어) 실험에서 합의한 것과 동일: **두 아키텍처에 동일한 생성 예산을 줘서
Diffusion의 denoise-budget 비대칭을 제거**한다.

| | **AR (Qwen)** | **Diffusion (Fast-dLLM v2)** |
|---|---|---|
| 디코딩 | autoregressive greedy (`do_sample=False`) | block diffusion `mdm_sample` (block 32, threshold 0.9) |
| 생성 예산 (max_new_tokens) | **64** (천장) | **64** (천장, block 배수로 올림 `ceil(64/32)*32=64`) |
| 종료 조건 (자동 길이 조절) | EOS 토큰에서 **per-token 자연 정지**, 없으면 64에서 컷 | EOS(`stop_token=151645`)를 denoising 중 감지 → 그 뒤 pad 처리 + 남은 블록 조기 종료(`finished_flag.all()→break`). **per-block(32) granularity**, EOS 없으면 천장에서 컷 |
| 후처리 (`clean_pred`) | 첫 비어있지 않은 줄 (label 접두사 제거) | 동일 |

- **양쪽 모두 길이를 모델이 자율 결정** — Diffusion은 캔버스를 강제로 다 채우지 않고 EOS 위치까지만
  생성 후 패딩(`fastdllm_generation.py` L125–166). max_new_tokens는 강제 길이가 아니라 **천장**.
  차이는 granularity뿐: AR=토큰 단위 정지, Diffusion=EOS가 나온 블록(32토큰)까지 생성 후 정지.
- **동일 예산(64)** → 어느 쪽도 길이 이점이 없음. test 출력 최대가 42토큰(§3.5)이라 64는 충분.
- **GT-length(오라클) 모드 — 본 평가에선 제거됨**: 선행 핀란드어 실험에선 `free`(첫 EOS 컷)와
  `gt_length`(EOS 스트립 후 정답 토큰 길이로 캡) **두 모드 모두** 측정했다. 인공어 평가(`eval_ft.py`)는
  *methodology audit* 결과 **`free`만** 사용 — gt_length는 정답 길이를 모델에 흘리는 오라클이라
  "둘 다 정답 길이 모름" 대칭을 깬다고 판단했기 때문. (⚠️ 단 CLAUDE.md 하드룰 #2는 학습 모델에
  free·gt_length **둘 다** 요구 → §9 caveat 5에서 재논의, 추가 측정 여부 결정 대기.)
- 핀란드어와의 차이 요약: ① 예산 256→64(짧은 문장이라 무해), ② gt_length 모드 미측정, ③ 지표에
  exact-match·어순 τ 추가.

### 3.4 메트릭 정의

| 메트릭 | 정의 | 인공어에서의 의미 |
|---|---|---|
| **chrF** | 문자 n-gram F-score (`sacrebleu.corpus_chrf`, 기본 char 6-gram, β=2) | 접사·교착어 형태론에 강건 — Klingon 접사(`-pu'`,`vI-`,`-'a'`) 부분일치를 포착. 주력 지표. |
| **BLEU** | 단어 n-gram precision(≤4-gram) + brevity penalty (`corpus_bleu`) | 정확 일치에 엄격. 짧거나 degenerate(반복) 출력에 가혹 → Diffusion 붕괴를 강하게 페널티. |
| **Exact match (EM)** | 후처리 후 예측==정답 문자열인 비율(%) | 가장 엄격. 인공어에선 대체로 낮음(완전 일치는 드묾). |
| **어순 τ** (word-order Kendall τ) | 예측·정답 **둘 다에 등장한 단어들**만 추려, 그 단어들이 정답 순서대로 놓였는지 Kendall rank 상관. ≥2개 일치 단어가 있는 쌍에서만 계산. | **어휘와 분리된 순수 어순 신호** — H1(양방향 디코딩이 어순에 유리)의 핵심 지표. 1.0=완벽 순서, 0=무상관, 음수=역순. |

### 3.5 절단(truncation) 검증

max_len/max_new_tokens에 의해 학습·평가 데이터가 잘렸는지 실측 (Qwen2.5 토크나이저, `data/klingon`).

| 검증 대상 | cap | 실측 max | cap 초과 비율 |
|---|---|---|---|
| **학습** 전체 길이(프롬프트+응답) | 512 (`--max_len`) | FWD 224 / REV 223 | **0.00%** (양 방향, train 12,717) |
| **평가** 출력 길이 (test) | 64 (`--max_new_tokens`) | FWD 42 / REV 29 | **0.00%** (test 500) |
| Khalani 출력 | — | 10 | — |

→ **학습·평가 어디서도 절단 없음.** 512는 학습 최대(224)의 2배 이상, 64는 test 출력 최대(42)보다 큼.
(train에서 출력>64인 예가 FWD 4개·REV 1개 있으나 이는 학습 입력일 뿐 평가엔 test만 쓰고 test는 0개.)

---

## 4. 결과 — 축별 상세

### 4.1 Klingon en→tlh (순방향) — ✅ 완료

숫자: `outputs/klingon_ft_eval_ar.json` / `outputs/klingon_ft_eval_diffusion.json` (n=300).

| 메트릭 | AR (Qwen) | Diffusion (Fast-dLLM) | 우세 |
|---|:---:|:---:|:---:|
| chrF | **38.43** | 35.67 | AR (근소) |
| BLEU | **10.40** | 2.55 | AR (큰 차) |
| Exact match | 8.33 | **8.67** | 무승부 |
| **어순 τ** | 0.909 (n=77) | **0.936** (n=68) | **Diffusion** |

- HF 체크포인트: AR [klingon-en2tlh-qwen2.5-7b-fullft](https://huggingface.co/sungkwang2/klingon-en2tlh-qwen2.5-7b-fullft) ·
  Diffusion [klingon-en2tlh-fastdllm-v2-7b-fullft](https://huggingface.co/sungkwang2/klingon-en2tlh-fastdllm-v2-7b-fullft)
- 무학습 few-shot ICL에선 양쪽 chrF ~10–11 → **full-FT로 ~4배 상승**(AR 10.6→38.4), 인공어가
  12.7k쌍으로 학습됨을 확인.
- **H1 약한 지지**: 어순 τ에서 Diffusion(0.936) > AR(0.909). 전체 유창성(chrF/BLEU)은 AR 우세.

### 4.2 Klingon tlh→en (역방향) — ⏳ 학습 완료, 평가 대기

| 메트릭 | AR (Qwen) | Diffusion (Fast-dLLM) |
|---|:---:|:---:|
| chrF | ⏳ TBD | ⏳ TBD |
| BLEU | ⏳ TBD | ⏳ TBD |
| Exact match | ⏳ TBD | ⏳ TBD |
| 어순 τ | ⏳ TBD | ⏳ TBD |

- AR 학습 완료(`klingon_qwen_tlh2en_fullft`): final train loss **1.148**, eval loss **1.234** (순방향 AR 1.044/1.129보다 약간 높음 — tlh 이해가 tlh 생성보다 어려운 신호일 수 있음, 평가로 확인 예정).
- Diffusion 학습 중(`klingon_fastdllm_tlh2en_fullft`, 1,590 스텝) → 완료 즉시 양쪽 `eval_ft.py --tgt_field en --lang_name English --test_file data/klingon/test_tlh2en.jsonl` 실행.
- HF 업로드: ⬜ TBD (평가 통과 후).

### 4.3 Khalani en→kha (순방향) — ✅ 완료 (탐색적, 과적합 실측)

> Khalani는 무명 인공어라 H1의 "미지 언어" 전제엔 가장 깨끗하지만 **번역쌍이 55개뿐**이라
> FT 결론은 낼 수 없음. 과적합을 실측 확인하는 목적의 탐색적 런.

fold0 = train 44 / test 11, 양쪽 full-FT, lr 2e-5, eff.batch 8, **epochs 20**.
숫자: `outputs/khalani_ft_eval_{ar_train,ar_test,diffusion_test}.json`.

| | AR train (44) | AR test (11) | Diffusion test (11) |
|---|:---:|:---:|:---:|
| chrF | **100.0** | 13.20 | 14.94 |
| BLEU | 100.0 | 7.68 | 0.00 |
| EM | **100%** | 0% | 0% |

- **AR이 학습셋 44개를 chrF/EM 100%로 완전 암기**하지만 held-out 11개에선 chrF 13·EM 0으로 붕괴
  → train↔test 격차 100→13 = **교과서적 과적합**. 55쌍 FT는 결론 불가임을 실증.
- held-out에선 양쪽 모두 실패(AR 13.2 vs DIFF 14.9 chrF), **n=11이라 차이는 노이즈**.

### 4.4 Khalani kha→en (역방향) — ⬜ 미착수

| 메트릭 | AR (Qwen) | Diffusion (Fast-dLLM) |
|---|:---:|:---:|
| 전체 | ⬜ TBD | ⬜ TBD |

- 역방향 Khalani 데이터(`data/khalani/*_en.jsonl`) 생성 ⬜ TBD.
- 단, 4.3에서 보듯 55쌍 규모로는 어느 방향도 결론 불가 — 우선순위 낮음 (Klingon 역방향 완료 후 결정).

---

## 5. Loss 곡선

![loss](figures/2026-05-28-klingon-loss.png)

| 런 | train loss | eval loss |
|---|---|---|
| Klingon en→tlh AR | 6.05 → **1.044** | 1.677 → **1.129** (단조 하강, 과적합 없음) |
| Klingon en→tlh Diffusion | 12.67 → **3.81** (블록 디퓨전 손실, 스케일 다름) | (디퓨저 트레이너엔 val 없음) |
| Klingon tlh→en AR | → **1.148** | → **1.234** |
| Klingon tlh→en Diffusion | ⏳ TBD (학습중) | (val 없음) |

- 그림은 현재 순방향 Klingon만 포함. **역방향 2개 추가한 통합 figure는 ⏳ TBD** (생성기 `reports/figures/2026-05-28-klingon-loss.py` 확장 예정).

---

## 6. 실제 예제

### 6.1 Klingon en→tlh (`outputs/klingon_ft_eval_*.json`의 `examples`)

| English | Reference (tlh) | AR | Diffusion |
|---|---|---|---|
| I hate lawyers. | `chut qeSwI'pu' vImuS.` | `chut qeSwI'pu' vImuS.` ✅정확 | `chut qeSwI' vImuS.` |
| Are you coming to the store with me? | `ngevwI'Daq vIghoStaHvIS chotlhej'a'?` | `jIghoStaH'a' chaHvaD beylIjDaq?` | `ngevwIj vIjaHmeH chutlhe'taH'a'?` |
| What do we draw? | `nuq wIDIj?` | `nuq wIHIv?` | `nuq wIlo' 'e' wIvang?` |
| My mouth forms a lot of saliva. | `tlhepQe' law' lIng nujwIj.` | `tlhoS jIbIjqu' 'oH jatlhlu'wIj'e'.` | `, we have 1, 2, 3, 4, 5, …` ⚠️degenerate |

- AR은 Klingon 형태론(접사 `-pu'`, `vI-`, `-'a'`)을 살린 문장 생성. Diffusion은 일부 입력에서
  `, we have 1,2,3,…` 반복 붕괴 → BLEU를 크게 깎음.

### 6.2 Klingon tlh→en (역방향) — ⏳ TBD
평가 완료 후 `outputs/klingon_tlh2en_ft_eval_*.json`의 `examples`에서 인용 예정.

### 6.3 Khalani en→kha (`outputs/khalani_ft_eval_ar_test.json`)
- `Prismatic core online` → ref `Peradak kural`, AR `Peradak aghanizha` (첫 단어만 맞고 뒤는 다른 암기 어휘).

---

## 7. 추론 스크립트

방향은 instruction 접두사만 다르다 (`Translate to Klingon:` ↔ `Translate to English:`).

### 7.1 AR (Qwen2.5-7B) — 표준 generate
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mid = "sungkwang2/klingon-en2tlh-qwen2.5-7b-fullft"   # 역방향 모델은 TBD 업로드
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="auto").eval()

def translate(text, instr="Translate to Klingon:"):
    msgs = [{"role": "user", "content": f"{instr} {text}"}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(p, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

print(translate("I hate lawyers."))   # -> chut qeSwI'pu' vImuS.
```

### 7.2 Diffusion (Fast-dLLM v2 7B) — 블록 디퓨전 샘플링
`fastdllm_generation.py`(이 repo, `eval/`)와 **transformers 4.x**(5.x는 생성이 깨짐)가 필요.
가장 간단한 재현은 repo의 `eval_ft.py`:
```bash
PYTHONPATH=eval python3 eval_ft.py --model_type diffusion \
  --model_path sungkwang2/klingon-en2tlh-fastdllm-v2-7b-fullft \
  --test_file data/klingon/test.jsonl --n_eval 10
```
핵심 호출(요약):
```python
import types, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import fastdllm_generation as gf  # repo 파일 (rope/DynamicCache 호환 패치 포함)

mid = "sungkwang2/klingon-en2tlh-fastdllm-v2-7b-fullft"
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True).eval()
model.mdm_sample = types.MethodType(gf.Fast_dLLM_QwenForCausalLM.batch_sample, model)
mask_id = tok.encode("|<MASK>|", add_special_tokens=False)[0]

msgs = [{"role": "user", "content": "Translate to Klingon: I hate lawyers."}]
p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = tok(p, return_tensors="pt").input_ids.to(model.device)
out = model.mdm_sample(ids, tokenizer=tok, block_size=32, small_block_size=32,
        max_new_tokens=64, mask_id=mask_id, min_len=ids.shape[1],
        seq_len=torch.tensor([ids.shape[1]], device=model.device),
        use_block_cache=True, threshold=0.9)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip())
```

---

## 8. 종합 해석

- **순방향 Klingon (유일하게 결론 가능한 축)**: 혼합·메트릭 의존적. **H1 약한 지지** — 어순 τ에서
  Diffusion(0.936) > AR(0.909)로 양방향 디코딩이 어순에 미세 우위. 단 전체 유창성(chrF/BLEU)은 AR 우세
  (Diffusion의 degenerate 반복 모드가 일부 입력에서 n-gram 정밀도를 무너뜨림). **깨끗한 승부는 아님.**
- **역방향 Klingon**: ⏳ 평가 대기. 출력이 영어(모델이 이미 아는 언어)라 어순 τ의 의미가 약해지고,
  H1 효과가 "생성"이 아니라 "이해"에서도 나타나는지 보는 보조 축.
- **Khalani (양 방향)**: 55쌍 규모가 근본 한계. 순방향에서 AR이 train 100% / test 13%로 **교과서적
  과적합** 실측 → AR/Diffusion 우열을 가릴 수 없음. "소량 인공어 FT = 암기" 사례로만 기록.

**결론**: 신뢰할 만한 AR-vs-Diffusion 결론은 현재 **Klingon 순방향** 한 축에서만 가능하며 H1을 약하게
지지한다. 역방향 Klingon 결과가 채워지면 "이해 vs 생성" 분해가 가능해진다.

---

## 9. 주의 (caveat)

1. Klingon은 온라인 자료가 많아 **Qwen 사전학습에 포함됐을 가능성** → 부분적으로 *회상*이지 순수
   *학습*이 아님(H1의 "미지 언어" 전제 오염). 더 깨끗한 미지 축은 Khalani(단, 55쌍이라 FT엔 부적합).
2. single seed, Diffusion은 샘플링 비결정적 → 작은 τ 격차 신뢰엔 ≥3 seed 필요.
3. 1 epoch, Diffusion 디코딩 하이퍼파라미터(block_size/threshold) 미튜닝.
4. 어순 τ는 ≥2개 단어가 일치한 test 쌍에서만 계산 → 검정력 보통. 역방향(영어 출력)에선 어순 τ 해석 주의.
5. **gt_length 모드 미측정**: 본 평가는 `free`만(§3.3). CLAUDE.md 하드룰 #2는 학습 모델에 free·gt_length
   둘 다 요구하므로, 엄밀 비교를 위해 양 모드 측정이 필요할 수 있음 → 추가 여부 사용자 결정 대기.

---

## 10. 재현

환경: `transformers 4.57.6`, `trl 1.4.0`, `torchao 0.17.0`, `peft 0.19.1`. 모델·데이터는 `/content/local_fast`에서만 읽음 (Drive는 백업 전용).

```bash
# --- 데이터 ---
python3 prepare_klingon.py                       # 순방향 en→tlh
# 역방향 tlh→en: 순방향 jsonl의 instruction/output 스왑 → data/klingon/*_tlh2en.jsonl

# --- 학습 (순방향 예시; 역방향은 train_file만 *_tlh2en.jsonl 로) ---
# AR
python3 train/train_qwen_v4.py --train_file data/klingon/train.jsonl --val_file data/klingon/val.jsonl \
  --lora_rank 0 --lr 2e-5 --epochs 1 --output_dir outputs/klingon_qwen_fullft
# Diffusion
python3 train/train_fastdllm.py --train_file data/klingon/train.jsonl --output_dir outputs/klingon_fastdllm_fullft \
  --lora_rank 0 --lr 2e-5 --epochs 1 --batch_size 1 --grad_accum 8 --max_len 512

# --- 평가 (방향별 --tgt_field / --lang_name / --test_file 만 교체) ---
PYTHONPATH=eval python3 eval_ft.py --model_type ar \
  --model_path outputs/klingon_qwen_fullft/final --n_eval 300 --out outputs/klingon_ft_eval_ar.json
PYTHONPATH=eval python3 eval_ft.py --model_type diffusion \
  --model_path outputs/klingon_fastdllm_fullft/final --n_eval 300 --out outputs/klingon_ft_eval_diffusion.json
```
