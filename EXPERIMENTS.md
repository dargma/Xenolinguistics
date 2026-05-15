# Experiments — AR LLM vs Diffusion LLM, en→fi

> AI-assistant log. Append-only results table; checkbox roadmap. Commands
> live in [`INSTALL.md`](INSTALL.md). Writing conventions live in
> [`reports/REPORT_GUIDE.md`](reports/REPORT_GUIDE.md).

---

## Results log (append-only)

Eval set: `data/test_1k.jsonl` first 100 sentences, sacrebleu defaults.
Diffusion LLM rows are `--mode gt_length` (oracle target length) unless noted.

| # | Model | Method | Data | Eval mode | chrF | BLEU | Artifact | Report |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| R0 | `opus-mt-tc-big-en-fi` | NMT, no FT | — | beam=4 | 56.91 | 37.45 | `outputs/reference/` | — |
| Q1 | Qwen2.5-7B-Instruct + LoRA r=16 | q/k/v/o, lr=2e-4, 3 ep | 100k | greedy free | 17.05 | 1.03 | `outputs/qwen_100k/adapter/` | — |
| F1 | Fast-dLLM v2 7B | Full FT, lr=2e-5, 1 ep | 10k | `free` | 38.59 | 18.98 | `outputs/fastdllm_v2_10k/final/` | — |
| F1' | Fast-dLLM v2 7B | Full FT, lr=2e-5, 1 ep | 10k | `gt_length` | TBD | TBD | `outputs/fastdllm_v2_10k/final/` | pending re-eval |

Append-only. New rows go below. Do not silently overwrite an existing row.
(Headline-number propagation rule lives in [`CLAUDE.md`](CLAUDE.md) §3.6.)

Caveats on existing rows:
- **Q1** trained 3 epochs on 100k and overfits. A prior 10k × 3 ep run reported
  chrF 41.72 / BLEU 22.89; that artifact was removed in 2026-05-15 cleanup
  and must be re-run before being cited.
- **F1** used `--mode free`; the headline number for Diffusion LLM is `gt_length`.

---

## Roadmap (checkbox state)

### Phase A — same-data baseline (10k)
- [x] Qwen LoRA r=16 q/k/v/o, 10k × 3 ep (artifact removed — re-run pending)
- [x] Fast-dLLM v2 Full FT, 10k × 1 ep
- [ ] Fast-dLLM v2 LoRA r=64 (all-linear), 10k

### Phase B — scale (100k)
- [x] Qwen LoRA r=16, 100k × 3 ep → overfits (chrF 17.05)
- [ ] Qwen LoRA r=16, 100k × **1 ep**
- [ ] Qwen LoRA r=64, 100k × 1 ep
- [ ] Fast-dLLM v2 LoRA r=64, 100k × 1 ep
- [ ] Fast-dLLM v2 Full FT, 100k × 1 ep

### Phase C — method ablation
- [ ] LoRA r ∈ {16, 32, 64} for both models, fixed data
- [ ] Full FT vs best LoRA per model

### Phase D — error analysis (required deliverable)

For the Phase A best Qwen run and best Fast-dLLM run, cover ≥3 axes from
[`reports/REPORT_GUIDE.md`](reports/REPORT_GUIDE.md) §Error-analysis axes.
Save plots under `reports/figures/`; save generator scripts alongside.

---

## Eval protocol — apply exactly, no improvisation

### Qwen (AR LLM)
- Chat template: `<|im_start|>user\n{instr}<|im_end|>\n<|im_start|>assistant\n`
- Decode: greedy, `max_new_tokens=128`, `do_sample=False`
- Script: `eval_qwen.py`

### Fast-dLLM v2 (Diffusion LLM)
- Decoder: `mdm_sample` (block-wise iterative denoising, confidence-threshold unmasking)
- `block_size=32`, `small_block_size=32`, `threshold=0.9`
- `--mode gt_length` (default): `max_new_tokens = ceil(len_tok(fi_ref) / block_size) · block_size`. Standard for Diffusion LLM translation eval (LLaDA, Dream, DiffusionLLM)
- `--mode free`: `max_new_tokens=256` cap. Matches Fast-dLLM v2 model-card defaults. Supplementary number, not headline
- Script: `eval_fastdllm.py`

### Common
- `data/test_1k.jsonl` first 100 sentences
- `sacrebleu.corpus_chrf` + `sacrebleu.corpus_bleu` (defaults)
- `examples` field: first 10 `(en, ref, pred)` tuples saved with each `eval_results.json`

---

## Comparison axes (what the experiment exists to measure)

| Axis | Qwen2.5-7B-Instruct (AR LLM) | Fast-dLLM v2 7B (Diffusion LLM) |
|---|---|---|
| Generation | autoregressive, left-to-right, EOS-terminated | block masked diffusion (`bd_size=32`) |
| Training objective | causal LM cross-entropy | block diffusion mask + complementary mask + token shift |
| Inference | greedy / beam | iterative denoising, confidence threshold |
| Base | Qwen2.5-7B | Qwen2.5-7B family |

Same base → quality differences trace to the generation paradigm, not pretraining data.
