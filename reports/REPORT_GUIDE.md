# 보고서 작성 가이드

> 한 런 = 한 보고서 (CLAUDE.md §3). 타인 보고서는 새 파일로 supersede.

## 위치

```
reports/
├── REPORT_GUIDE.md          # 이 파일
├── figures/                 # 평탄: YYYY-MM-DD-<short>-<what>.{png,svg}
└── <user_id>/               # HF 핸들 또는 github id (소문자)
    └── YYYY-MM-DD-<short>.md
```

## 헤더

```markdown
# YYYY-MM-DD — <짧은 제목>

| 필드 | 값 |
|---|---|
| Author | <user_id> |
| Run dir | `outputs/<run>/` |
| Git commit | `<hash>` |
```

## 본문 필수 3 요소

1. **Loss 곡선** — 학습 loss (있으면 val loss도) vs step.
   - Qwen: `outputs/<run>/log_history.json` (`loss` / `eval_loss` 필드)
   - Fast-dLLM v2: `outputs/<run>/checkpoint-*/trainer_state.json` 또는 `train.log` 의 `'loss': ...` 라인
   - 그림 저장: `reports/figures/<YYYY-MM-DD>-<short>-loss.png` + 생성 스크립트 옆에.

2. **평가 방법론** — 사용 스크립트, `--mode`, 정확한 CLI 명령, `n_eval`, 디코딩 설정. 한 단락 또는 표.

3. **재현 라인** — 학습 명령 + 평가 명령 + git commit 해시 (그대로 복붙 가능해야 함).

## 에러 분석 (AR vs Diffusion 비교 시 ≥ 3 축 선택)

| 축 | 측정 |
|---|---|
| 길이 | `|pred_tokens| - |gt_tokens|` 분포 |
| 형태론 | 대상 언어의 굴절·격·일치 오류 (수동, n ≥ 30) |
| 반복 | 2-/3-gram 반복률 (정규화) |
| 안정성 | 동일 prompt 5 seed의 chrF 분산 |
| 충실도 | 환각 / 누락 span (수동, n ≥ 20) |

## 그림 규약

- 축 라벨 + 범례 필수, 150 DPI, 8 × 4 인치 기본
- 색: 베이스라인 회색, AR LLM 파랑, Diffusion LLM 주황

## 수치 표기

chrF/BLEU 소수 **2자리** · loss 소수 **4자리** · wall time `M:SS` < 60 m 그 외 `H:MM` · VRAM `X.X GB`
