# Xenolinguistics — AR LLM vs Diffusion LLM, en→fi

Apple-to-apple comparison of an **AR LLM** (`Qwen/Qwen2.5-7B-Instruct`)
and a **Diffusion LLM** (`Efficient-Large-Model/Fast_dLLM_v2_7B`) on
English → Finnish translation (OPUS-100). Same base, same data, same eval.

## Map

| File | Purpose |
|---|---|
| [`INSTALL.md`](INSTALL.md) | Install + commands. Single source for reproduction. |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Results log + roadmap + eval protocol. |
| [`CLAUDE.md`](CLAUDE.md) | Hard rules for AI assistants. |
| [`reports/REPORT_GUIDE.md`](reports/REPORT_GUIDE.md) | How to log a new run. |

## Current headline (2026-05-15)

Eval: `data/test_1k.jsonl` first 100, sacrebleu defaults.
Diffusion LLM uses oracle target length (`--mode gt_length`).

| Model | Method | Data | chrF | BLEU |
|---|---|:---:|:---:|:---:|
| `opus-mt-tc-big-en-fi` | NMT baseline | — | **56.91** | **37.45** |
| Qwen2.5-7B + LoRA r=16 | q/k/v/o, 3 ep | 100k | 17.05 | 1.03 |
| Fast-dLLM v2 7B Full FT | 1 ep | 10k | 38.59 | 18.98 |

Caveats: Qwen 100k × 3 ep overfits; Fast-dLLM number used `--mode free`,
re-eval under `gt_length` pending. Details in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## References

- Fast-dLLM v2 7B — https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B
- Qwen2.5-7B-Instruct — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- OPUS-100 — https://huggingface.co/datasets/Helsinki-NLP/opus-100
- Reference NMT — https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-fi
