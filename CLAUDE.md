# AI assistant rules

> Hard rules only. Commands + results → [`README.md`](README.md).
> Reports → [`reports/REPORT_GUIDE.md`](reports/REPORT_GUIDE.md).

## Fixed model roles

| Role | HF id |
|---|---|
| Reference NMT (never train) | `Helsinki-NLP/opus-mt-tc-big-en-fi` |
| AR LLM | `Qwen/Qwen2.5-7B-Instruct` |
| Diffusion LLM | `Efficient-Large-Model/Fast_dLLM_v2_7B` |

Changing the comparison → update this table first.

## Hard rules

1. Each model reports **both** `free` and `gt_length` eval modes. Do not drop one.
2. One run = one `outputs/<run>/` directory. Contents:
   `eval_free.json`, `eval_gt_length.json`, `train.log`, plus the model
   weights subdir (`final/` or `checkpoint-N/`). Nothing else.
3. One run = one report under `reports/<user_id>/YYYY-MM-DD-<short>.md`.
4. Never delete or edit someone else's report. Supersede with a new file.
5. Numbers come from JSON files, never from chat or memory.
6. New headline number → update the row in [`README.md`](README.md) §1 Results.

## Anti-patterns

- ❌ Adding new top-level files for one-off scripts. Extend an existing one.
- ❌ Editing weights in `outputs/` by hand.
- ❌ Writing a "summary" markdown that paraphrases `eval_{free,gt_length}.json`. Open the JSON.
- ❌ Compatibility patches outside `train/train_fastdllm.py` / `eval/fastdllm_generation.py` / `train/train_qwen_v4.py` / `eval/eval_*.py`.
- ❌ Silent fallbacks (`try: ... except: pass`) around model loading. Errors should be loud.
- ❌ Mixing modes in a single comparison row. Compare `free` vs `free` and `gt_length` vs `gt_length` only.

## Comparison checklist

Before claiming any comparison:
1. Both `eval_free.json` and `eval_gt_length.json` opened. Numbers cited with path.
2. Same `test_file` and `n_eval` in both.
3. Same eval mode on both sides being compared.
4. Qwen used the canonical chat template
   (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`).
