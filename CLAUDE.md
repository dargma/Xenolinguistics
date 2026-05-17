# AI 어시스턴트 규칙

> 하드 룰만. 명령어 + 결과 → [`README.md`](README.md).
> 보고서 → [`reports/REPORT_GUIDE.md`](reports/REPORT_GUIDE.md).

## 고정 모델 역할

| 역할 | HF id |
|---|---|
| Reference NMT (학습 금지) | `Helsinki-NLP/opus-mt-tc-big-en-fi` |
| AR LLM | `Qwen/Qwen2.5-7B-Instruct` |
| Diffusion LLM | `Efficient-Large-Model/Fast_dLLM_v2_7B` |

비교 대상을 바꾸려면 이 표부터 갱신.

## 하드 룰

1. 각 모델은 `free` 와 `gt_length` **두 모드 모두** 보고. 한쪽만 보고하지 말 것.
2. 런 하나 = `outputs/<run>/` 하나. 구성: `eval_free.json`, `eval_gt_length.json`,
   `train.log`, 모델 weight 서브디렉터리(`final/` 또는 `checkpoint-N/`). 그 외 금지.
3. 런 하나 = 보고서 하나 (`reports/<user_id>/YYYY-MM-DD-<short>.md`).
4. 타인의 보고서는 절대 삭제·수정 금지. 새 파일로 supersede.
5. 숫자는 JSON에서. 대화·기억에서 가져오지 말 것.
6. 새 headline 숫자 → [`README.md`](README.md) §1 표 갱신.

## 안티패턴

- ❌ 일회용 스크립트를 위한 최상위 파일 추가. 기존 파일 확장으로.
- ❌ `outputs/` 안 weight를 손으로 편집.
- ❌ `eval_{free,gt_length}.json` 을 그대로 옮겨 적는 "요약" 마크다운. JSON을 직접 열 것.
- ❌ 호환성 패치를 `train/train_fastdllm.py` / `eval/fastdllm_generation.py` /
  `train/train_qwen_v4.py` / `eval/eval_*.py` 밖에 두는 행위.
- ❌ 모델 로딩에 침묵 fallback (`try: ... except: pass`). 에러는 시끄럽게.
- ❌ 한 비교 행 안에서 모드 혼합. `free` vs `free`, `gt_length` vs `gt_length` 만.

## 비교 체크리스트

비교 주장 전에 확인:
1. `eval_free.json` 과 `eval_gt_length.json` 을 직접 열어 경로와 함께 인용.
2. 양쪽이 같은 `test_file` 과 `n_eval`.
3. 양쪽이 같은 평가 모드.
4. Qwen은 표준 chat template 사용
   (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`).
