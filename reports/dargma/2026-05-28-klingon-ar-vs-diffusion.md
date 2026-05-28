# 2026-05-28 — Klingon (conlang) AR vs Diffusion, full fine-tune

| Field | Value |
|---|---|
| Author | dargma |
| Date | 2026-05-28 |
| Run dir | `outputs/klingon_qwen_fullft/`, `outputs/klingon_fastdllm_fullft/` |
| Git commit | `0709965` |
| AR checkpoint (HF) | https://huggingface.co/sungkwang2/klingon-en2tlh-qwen2.5-7b-fullft |
| Diffusion checkpoint (HF) | https://huggingface.co/sungkwang2/klingon-en2tlh-fastdllm-v2-7b-fullft |

## Goal / hypothesis

Does LLM **architecture type** help learn a constructed/alien language? **H1:** a conlang differs
from English in vocabulary *and word order* and is (largely) absent from pretraining; a Diffusion
LLM (bidirectional denoising) should learn it — especially word order — better than a left-to-right
AR LLM. This report covers the **Klingon** axis (OVS word order; real sentence corpus). Khalani is a
separate, much smaller axis.

## Data

- Source: OPUS **Tatoeba en-tlh** v2023-04-12 (CC-BY). Dedup exact pairs → **13,717** unique.
- Split (`data/klingon/`): train 12,717 / val 500 / test 500. Built by `prepare_klingon.py`.
- Direction: **en→klingon** (generation exposes word order).

## Models / training

| | AR | Diffusion |
|---|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` | `Efficient-Large-Model/Fast_dLLM_v2_7B` |
| Method | **full fine-tune** | **full fine-tune** |
| LR | 2e-5 | 2e-5 |
| Epochs | 1 | 1 |
| Eff. batch | 8 | 8 |

Both use **identical** lr / epochs / effective batch for a fair comparison. LoRA was rejected:
prior en→fi evidence shows Fast-dLLM barely learns under LoRA (any rank), so LoRA would handicap
the diffusion side — full-FT is the only fair common ground.

## Loss curve

![loss](../figures/2026-05-28-klingon-loss.png)

- **AR (Qwen):** train 6.05 → 1.04; **eval_loss 1.677 → 1.129** (monotone over 7 points) → genuine
  generalization, no overfit at 1 epoch. Stable grad-norm ~15–18.
- **Diffusion (Fast-dLLM):** train (block-diffusion loss, different scale) 12.67 → 3.81, steady
  decline. No val set in the diffusion trainer.
- Points captured from the training logs; per-step `trainer_state.json` was pruned by
  `save_total_limit=1` + disk cleanup. Generator: `reports/figures/2026-05-28-klingon-loss.py`.

## Eval methodology

Script `eval_ft.py`, **0-shot** (instruction only), **symmetric FREE-length for both** (the
`gt_length` oracle was removed so neither model gets the reference length). `n_eval=300` on
`data/klingon/test.jsonl`. AR = greedy (`do_sample=False`); Diffusion = `mdm_sample`, block_size 32,
threshold 0.9, free length. Metrics via `sacrebleu` (chrF, BLEU) + exact-match + **word-order τ**
(Kendall τ on the order of reference-matched words — a vocabulary-decoupled word-order signal).

Commands:
```
HF_HOME=/content/local_fast/hf_cache python3 eval_ft.py --model_type ar \
  --model_path outputs/klingon_qwen_fullft/final --n_eval 300 --out outputs/klingon_ft_eval_ar.json
HF_HOME=/content/local_fast/hf_cache python3 eval_ft.py --model_type diffusion \
  --model_path outputs/klingon_fastdllm_fullft/final --n_eval 300 --out outputs/klingon_ft_eval_diffusion.json
```

## Results

Numbers from `outputs/klingon_ft_eval_ar.json` and `outputs/klingon_ft_eval_diffusion.json` (n=300).

| Metric | AR (Qwen) | Diffusion (Fast-dLLM) | Winner |
|---|:---:|:---:|:---:|
| chrF | **38.43** | 35.67 | AR (slight) |
| BLEU | **10.40** | 2.55 | AR (large) |
| Exact match | 8.33 | **8.67** | tie |
| **word-order τ** | 0.909 (n=77) | **0.936** (n=68) | **Diffusion** |

For reference, few-shot ICL (no training) scored chrF ~10–11 for both → **full-FT gives a ~4× jump**
(AR 10.6 → 38.4), confirming the conlang is learned from the 12.7k pairs.

## Interpretation

Mixed / metric-dependent:
- **Weak support for H1.** On the H1-relevant metric (word-order τ) **Diffusion (0.936) > AR (0.909)**:
  when the diffusion model produces the right words it orders them slightly more like the reference,
  consistent with bidirectional decoding helping word order.
- **AR wins overall fluency/precision** (BLEU 10.4 vs 2.55; chrF 38.4 vs 35.7). The BLEU gap is driven
  by **diffusion degenerate modes** on some inputs (e.g. emitting `, we have 1, 2, 3, 4, …`), which
  destroy n-gram precision.
- Net: no clean winner; diffusion shows a faint word-order edge but is less robust.

## Caveats

1. **Klingon is likely in Qwen pretraining** (well-documented online) → partly *recall*, not pure
   *learning*; contaminates the "unseen language" half of H1. (Khalani is the cleaner unseen testbed.)
2. Single seed; diffusion sampling is non-deterministic → ≥3 seeds needed to trust the small τ gap.
3. 1 epoch; diffusion degenerate modes untuned (block_size/threshold defaults).
4. word-order τ computed on only 68–77 test pairs with ≥2 matched words → modest power.

## Reproducibility

Env: `transformers==4.53.1`, `trl==0.19.1`, `torchao==0.17.0`, `peft 0.19.1` (see `INSTALL.md`).
Git commit `0709965`.
```
python3 prepare_klingon.py
# AR
HF_HOME=/content/local_fast/hf_cache python3 train_qwen_v4.py \
  --train_file data/klingon/train.jsonl --val_file data/klingon/val.jsonl \
  --lora_rank 0 --lr 2e-5 --epochs 1 --batch_size 8 --grad_accum 1 \
  --output_dir outputs/klingon_qwen_fullft
# Diffusion
HF_HOME=/content/local_fast/hf_cache python3 train_fastdllm.py \
  --train_file data/klingon/train.jsonl --output_dir outputs/klingon_fastdllm_fullft \
  --lora_rank 0 --lr 2e-5 --epochs 1 --batch_size 4 --grad_accum 2
```
