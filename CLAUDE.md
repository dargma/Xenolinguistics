# AI 어시스턴트 규칙

> 명령어·결과·환경 → [`README.md`](README.md)
> 보고서 작성 규약 → [`reports/REPORT_GUIDE.md`](reports/REPORT_GUIDE.md)

---

## 고정 모델

| 역할 | HF id |
|---|---|
| Reference NMT (학습 금지) | `Helsinki-NLP/opus-mt-tc-big-en-fi` |
| AR LLM | `Qwen/Qwen2.5-7B-Instruct` |
| Diffusion LLM | `Efficient-Large-Model/Fast_dLLM_v2_7B` |

→ 비교 대상을 바꾸려면 이 표를 먼저 갱신.

---

## 하드 룰

### 1. 숫자는 JSON에서
- 결과 인용 시 반드시 `eval_*.json` 경로와 함께.
- 옮겨 적은 "요약" 마크다운 금지 — 원본 JSON을 열 것.

### 2. 두 모드 측정
- 학습된 모델: `free` · `gt_length` **모두** 측정.
- 비교는 **같은 모드끼리만** (`free` vs `free`, `gt_length` vs `gt_length`).
- 베이스라인(opus-mt): `free`만.

### 3. 한 런 = 한 디렉터리 = 한 보고서
- 산출물: `outputs/<run>/`
- 보고서: `reports/<user_id>/YYYY-MM-DD-<short>.md`
- 타인 보고서 수정·삭제 금지. 갱신은 새 파일로 supersede.

### 4. 헤드라인 갱신
- 새 headline 숫자 → [`README.md`](README.md) §1 표를 즉시 갱신.

### 5. 코드 위치 고정
- 호환성 패치는 `train/` · `eval/` 안에서만.
- 최상위에 일회용 스크립트를 추가하지 말 것 (기존 파일 확장).

### 6. Git 협업
- `main` 직접 push 금지. `feat/<short>`, `fix/<short>` 등 브랜치에서 작업 후 PR.
- 1 커밋 = 1 논리 단위. 학습 weight, 대용량 로그는 커밋 금지 (`.gitignore` 신뢰).
- 커밋 메시지: 1줄 제목(50자 이내) + 빈 줄 + 본문. `왜` 위주, `무엇`은 diff로 충분.
- 다른 사람 브랜치 force-push 금지.

**Merge 규약**:
- PR ≥ 1명 승인 후 머지. self-merge 금지.
- 머지 전 `main` 최신화: `git fetch origin && git rebase origin/main` (충돌 해결은 PR 작성자).
- 기본은 **squash merge** (린한 main 히스토리). 단계별 커밋 의미가 중요할 때만 rebase merge.
- 머지 직후 원격·로컬 브랜치 삭제 (`git push origin --delete <br>` → `git branch -d <br>`).
- Headline 숫자나 모델 역할 변경 → PR 제목에 `[headline]` / `[models]` 태그 (다른 AI가 README 갱신 누락 안 하도록).
