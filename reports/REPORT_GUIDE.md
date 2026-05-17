# 보고서 가이드

런 하나당 파일 하나. Append-only. 타인 보고서 수정 금지.

## 위치

```
reports/
├── REPORT_GUIDE.md
├── figures/                  # 평탄: YYYY-MM-DD-<short>-<what>.{png,svg}
└── <user_id>/                # HF 핸들 또는 github id (소문자)
    └── YYYY-MM-DD-<short>.md
```

## 필수 헤더

```markdown
# YYYY-MM-DD — <짧은 제목>

| 필드 | 값 |
|---|---|
| Author | <user_id> |
| Date | YYYY-MM-DD |
| Run dir | `outputs/<run>/` |
| Git commit | `<hash>` |
```

본문은 자유 형식이지만 **다음 3가지는 필수**:

1. **Loss 곡선** — 학습 loss (있으면 val loss도) vs step.
   - Qwen: `outputs/<run>/log_history.json` 의 `loss` / `eval_loss` 항목으로 플롯.
   - Fast-dLLM v2: `outputs/<run>/checkpoint-*/trainer_state.json` 또는 `train.log` 의 `'loss': ...` 라인.
   - 그림은 `reports/figures/<YYYY-MM-DD>-<short>-loss.png` 에, 생성 스크립트도 옆에.
2. **평가 방법론** — 어떤 스크립트, 어떤 `--mode`, 정확한 CLI 명령, `n_eval`, 디코딩 설정(greedy / threshold / max_new_tokens). 한 단락 또는 작은 표.
3. **재현 라인** — 정확한 학습 명령 + git commit 해시.

## 규칙

1. 모든 수치 주장은 `outputs/<run>/eval_{free,gt_length}.json` 인용.
2. 인용 전 JSON 재오픈. 기억·대화 이력 금지.
3. 학습/평가 명령 둘 다 포함 (재현 라인).
4. 그림은 `reports/figures/`. 생성 스크립트는 옆에.
5. 결론을 단독으로 쓰지 말 것 — 시드 분산 / 베이스라인 대비로 표현.
6. 새 headline 숫자 → [`CLAUDE.md`](../CLAUDE.md) §6 전파 규칙.

## 에러 분석 축 (AR vs Diffusion 비교 필수, ≥ 3개 선택)

| 축 | 측정 대상 |
|---|---|
| 길이 | `|pred_tokens| - |gt_tokens|` 분포 |
| 형태론 | 핀란드어 명사 격 / 동사 일치 오류 (수동, n ≥ 30) |
| 반복 | 2-gram / 3-gram 반복률(정규화) |
| 안정성 | 동일 prompt 5 seed의 chrF 분산 |
| 충실도 | 환각 vs 누락 span (수동, n ≥ 20) |

## 그림 규약

- 축 라벨 + 범례 필수, 150 DPI, 기본 8 × 4 인치
- 색: opus-mt 회색, Qwen 파랑, Fast-dLLM v2 주황

## 수치 표기

chrF/BLEU 소수 2자리 · loss 소수 4자리 · wall time `min` < 60 그 외 `H:MM` · VRAM `X.X GB`
