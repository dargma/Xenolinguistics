# AI assistant rules

> Hard rules only. Commands → [`INSTALL.md`](INSTALL.md).
> Results & roadmap → [`EXPERIMENTS.md`](EXPERIMENTS.md).
> Reports → [`reports/REPORT_GUIDE.md`](reports/REPORT_GUIDE.md).

## Fixed model roles

| Role | HF id |
|---|---|
| Reference NMT (never train) | `Helsinki-NLP/opus-mt-tc-big-en-fi` |
| AR LLM | `Qwen/Qwen2.5-7B-Instruct` |
| Diffusion LLM | `Efficient-Large-Model/Fast_dLLM_v2_7B` |

Changing the comparison → update this table first.

## Hard rules

1. Diffusion LLM eval defaults to `--mode gt_length` (oracle target length).
   `free` is supplementary, not headline.
2. One run = one `outputs/<run>/` directory. Weights + `eval_results.json` +
   `training_meta.json` + `train.log`. Nothing else.
3. One run = one report under `reports/<user_id>/YYYY-MM-DD-<short>.md`.
4. Never delete or edit someone else's report. Supersede with a new file.
5. Numbers come from JSON files, never from chat or memory.
6. New headline number → update [`README.md`](README.md) §Current headline
   **and** add a row to [`EXPERIMENTS.md`](EXPERIMENTS.md) §Results log.

## Anti-patterns

- ❌ Adding new top-level files for one-off scripts. Extend an existing one.
- ❌ Editing weights in `outputs/` by hand.
- ❌ Writing a "summary" markdown that paraphrases `eval_results.json`. Open the JSON.
- ❌ Compatibility patches outside `train_fastdllm.py` / `fastdllm_generation.py`.
- ❌ Silent fallbacks (`try: ... except: pass`) around model loading. Errors should be loud.
- ❌ Mixing AR/Diffusion eval modes: Qwen = greedy free-length, Fast-dLLM = `gt_length`. Asymmetry is intentional.

## Comparison checklist

Before claiming any comparison:
1. Both `eval_results.json` opened. Numbers cited with path.
2. Same `test_file` and `n_eval` in both.
3. Fast-dLLM ran with `mode=gt_length` (unless explicitly asked otherwise).
4. Qwen used the canonical chat template
   (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`).
