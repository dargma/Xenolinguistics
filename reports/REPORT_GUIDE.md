# Report guide

One file per run. Append-only. Never edit someone else's.

## Location

```
reports/
├── REPORT_GUIDE.md
├── figures/                  # flat: YYYY-MM-DD-<short>-<what>.{png,svg}
└── <user_id>/                # your HF handle or github id, lowercase
    └── YYYY-MM-DD-<short>.md
```

## Mandatory header

```markdown
# YYYY-MM-DD — <short title>

| Field | Value |
|---|---|
| Author | <user_id> |
| Date | YYYY-MM-DD |
| Run dir | `outputs/<run>/` |
| Git commit | `<hash>` |
```

Body after the header is free-form. Cover what matters for the run.

## Rules

1. Every numeric claim cites `outputs/<run>/eval_results.json`.
2. Re-open the JSON before quoting a number. No memory, no chat history.
3. Include the exact training and eval commands (the reproducibility line).
4. Figures go in `reports/figures/`; save the generator script next to them.
5. No bare conclusions — frame gaps relative to seed variation / baseline.
6. New headline number → see [`CLAUDE.md`](../CLAUDE.md) §6 propagation rule.

## Error-analysis axes (required for AR vs Diffusion comparisons, ≥ 3)

| Axis | What to measure |
|---|---|
| Length | `\|pred_tokens\| - \|gt_tokens\|` histogram |
| Morphology | Finnish noun-case / verb-agreement errors (manual, n ≥ 30) |
| Repetition | 2- and 3-gram repeat rate, normalized |
| Stability | chrF spread across 5 seeds on the same prompt |
| Faithfulness | Hallucinated vs dropped content spans (manual, n ≥ 20) |

## Figure conventions

- Axis labels + legend mandatory; 150 DPI; 8 × 4 in default
- Colors: opus-mt gray, Qwen blue, Fast-dLLM v2 orange

## Number formatting

chrF/BLEU 2 decimals · loss 4 decimals · wall time `min` < 60 else `H:MM` · VRAM `X.X GB`
