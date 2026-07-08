# 데이콘 풍력발전량 예측 AI 경진대회 초기 플랜

## 1. 기본 방향

이번 대회에서는 **GPT + Codex CLI + VS Code + Colab** 중심으로 개발한다.  
Claude Code는 사용하지 않는다.

최종 제출 코드는 반드시 재현 가능해야 하며, 학습 코드와 추론 코드를 분리한다.  
원격 API 기반 모델 추론은 제한될 수 있으므로 GPT, Codex CLI, Copilot 같은 AI 도구는 **개발 보조용**으로만 사용하고, 최종 `train` / `inference` 코드 안에는 넣지 않는다.

핵심 역할 분담은 다음과 같다.

- **GPT**: 전략 수립, 평가식 해석, 피처 아이디어, 오류 원인 분석, 실험 결과 판단
- **Codex CLI**: 실제 코드 작성/수정, 폴더 구조 생성, `.py` 리팩토링, 실험 자동화
- **VS Code**: 메인 개발 환경
- **VS Code Copilot**: 짧은 코드 자동완성 보조
- **Colab**: LightGBM, XGBoost, CatBoost 등 학습 실행
- **Notion MCP**: 선택사항. 실험 로그와 할 일 관리용
- **omx / yolo mode**: 반복 작업 자동화용. 단, git commit 후 작은 범위에서만 사용

---

## 2. 최종 작업 흐름

```text
GPT에서 전략/피처/검증 설계 논의
        ↓
Codex CLI로 코드 생성·수정
        ↓
VS Code에서 파일 확인
        ↓
Colab에서 학습 실행
        ↓
결과 점수/로그를 GPT에 붙여서 다음 실험 결정
        ↓
Codex CLI로 다음 실험 코드 반영
```

이 방식의 핵심은 **Codex CLI가 코드 수정 담당, GPT가 판단 담당**이라는 점이다.

---

## 3. 프로젝트 폴더 구조

`.claude/` 폴더는 만들지 않는다.  
대신 Codex CLI와 사람이 함께 읽을 수 있는 문서 폴더를 둔다.

```text
dacon-wind-bar2026/
├─ AGENTS.md
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/              # open.zip 압축해제, git 제외
│  ├─ interim/
│  └─ processed/
├─ notebooks/
│  ├─ 00_eda.ipynb
│  ├─ 01_baseline_rf.ipynb
│  ├─ 02_lgbm_cv.ipynb
│  └─ 03_ensemble.ipynb
├─ src/
│  └─ dacon_wind/
│     ├─ data.py
│     ├─ features.py
│     ├─ metric.py
│     ├─ cv.py
│     ├─ models.py
│     ├─ ensemble.py
│     └─ submit.py
├─ scripts/
│  ├─ make_features.py
│  ├─ train_lgbm.py
│  ├─ train_xgb.py
│  ├─ train_catboost.py
│  ├─ make_ensemble.py
│  └─ validate_submission.py
├─ configs/
│  ├─ lgbm.yaml
│  ├─ xgb.yaml
│  └─ catboost.yaml
├─ outputs/
│  ├─ models/
│  ├─ predictions/
│  ├─ logs/
│  └─ figures/
├─ submissions/
├─ docs/
│  ├─ rules.md
│  ├─ leakage_checklist.md
│  ├─ experiment_log.md
│  └─ feature_ideas.md
└─ prompts/
   ├─ codex_refactor.md
   ├─ codex_feature.md
   ├─ codex_model.md
   └─ codex_review.md
```

---

## 4. AGENTS.md에 넣을 규칙

`AGENTS.md`는 Codex CLI가 항상 지켜야 할 프로젝트 규칙을 적는 파일로 사용한다.

```md
# Dacon Wind BARAM 2026 Agent Rules

## Goal
Maximize official DACON score:
Score = 0.5 * (1-NMAE) + 0.5 * FICR.

## Hard Rules
- Use Python only.
- Do not use random train/validation split.
- Use time-based validation.
- Never use 2025 target values.
- Never use information generated after prediction reference time.
- Do not call OpenAI, Gemini, Claude, Hugging Face Inference API, OpenRouter, or any remote API inside train/inference code.
- Train code and inference code must be separated.
- All final outputs must be reproducible.
- All CSV files must be UTF-8 or UTF-8-SIG.
- Do not modify ID columns in sample_submission.csv.

## Modeling Rules
- Train separate models for kpx_group_1, kpx_group_2, kpx_group_3.
- Always calculate local validation score using official metric.
- Track total_score, one_minus_nmae, ficr separately.
- Clip negative predictions to 0.
- Clip upper predictions by group capacity if capacity is available.
- Prefer stable CV improvement over public leaderboard overfitting.

## Workflow
- Before large edits, inspect existing files.
- Change one experiment component at a time.
- Save every submission with experiment id.
- Log experiment result in outputs/logs/experiments.csv.
```

---

## 5. Codex CLI 작업 프롬프트 예시

Codex CLI에는 한 번에 하나의 작업만 시킨다.  
초기에는 아래 순서대로 진행한다.

### 5.1 프로젝트 구조 만들기

```text
Create the project structure for a DACON wind power forecasting competition.
Do not modify data files.
Create src/dacon_wind modules, scripts, docs, configs, outputs, and submissions folders.
Add AGENTS.md, README.md, requirements.txt, and .gitignore.
Keep the code minimal and reproducible.
```

### 5.2 베이스라인 노트북을 `.py`로 분리

```text
Refactor the baseline notebook into reusable Python modules.
Separate data loading, feature engineering, metric calculation, model training, and submission generation.
Create train and inference scripts separately.
Do not change the modeling logic yet.
```

### 5.3 공식 metric 구현

```text
Implement the official DACON metric in src/dacon_wind/metric.py.
It must calculate:
- group NMAE
- one_minus_nmae
- FICR
- total_score = 0.5 * one_minus_nmae + 0.5 * FICR
Use only rows where actual generation is at least 10 percent of capacity.
Add simple tests or sanity checks.
```

### 5.4 LightGBM 실험 추가

```text
Add a LightGBM training script with time-based validation.
Train separate models for each KPX group.
Save validation predictions, model files, feature importance, and experiment logs.
Do not create submission unless validation succeeds.
```

---

## 6. 초기 우선순위

1순위는 모델 성능이 아니라 **검증 체계 구축**이다.  
로컬 검증 점수가 믿을 수 있어야 이후 실험이 의미 있다.

| 순서 | 해야 할 일 | 목표 |
|---:|---|---|
| 1 | `open.zip` 압축해제 | 데이터 구조 파악 |
| 2 | 베이스라인 실행 | 첫 제출 가능 상태 확보 |
| 3 | 공식 metric 구현 | 로컬 점수 신뢰성 확보 |
| 4 | 2024 validation 고정 | Public leaderboard 과적합 방지 |
| 5 | LightGBM 1차 모델 | RandomForest보다 강한 기본 모델 확보 |
| 6 | 풍속/풍향/시간 피처 추가 | 점수 상승 시도 |
| 7 | LGBM/XGB/CatBoost 앙상블 | 상위권 진입 시도 |
| 8 | 제출파일 검증 스크립트 | 제출 실수 방지 |

---

## 7. 도구 사용 규칙

### 7.1 GPT

GPT는 직접 코드를 많이 쓰기보다 다음 판단에 사용한다.

- 대회 규칙 해석
- 평가식 이해
- 검증 전략 설계
- 피처 아이디어 제안
- 로그/오류 원인 분석
- 실험 결과 비교
- 다음 실험 우선순위 결정

### 7.2 Codex CLI

Codex CLI는 실제 코드 수정을 담당한다.

단, 한 번에 너무 큰 작업을 시키지 않는다.

나쁜 예시:

```text
metric 구현하고 LGBM 추가하고 EDA까지 다 해줘.
```

좋은 예시:

```text
Implement the official metric only.
Do not modify training scripts.
Add minimal sanity checks.
```

### 7.3 VS Code

VS Code에서는 다음을 확인한다.

- 파일 구조
- Git 변경사항
- 코드 실행 오류
- `.env`, `.gitignore`, data 경로
- 제출 파일 위치

### 7.4 Colab

Colab은 학습 실행용으로 사용한다.

- LightGBM 학습
- XGBoost 학습
- CatBoost 학습
- 앙상블 실험
- 장시간 실행이 필요한 실험

Colab에서 실행한 결과는 반드시 다음 형태로 저장한다.

- validation prediction
- model file
- feature importance
- experiment log
- submission csv

### 7.5 VS Code Copilot

Copilot은 짧은 자동완성에만 사용한다.

핵심 로직은 반드시 Codex CLI 또는 GPT로 검토한다.

### 7.6 yolo mode / omx

초반에는 사용하지 않는다.  
프로젝트 구조가 안정된 뒤 반복 작업에만 사용한다.

사용해도 되는 예시:

- 코드 포맷 정리
- 실험 로그 컬럼 추가
- README 갱신
- docstring 추가
- 단순 import 정리

사용하면 위험한 예시:

- metric 수정
- validation split 수정
- inference 로직 수정
- submission 생성 로직 수정
- 데이터 누수 가능성이 있는 피처 추가

---

## 8. Git 작업 규칙

Codex CLI 작업 전에는 항상 현재 상태를 저장한다.

```bash
git status
git add .
git commit -m "baseline before metric implementation"
```

Codex CLI 작업 후에는 변경사항을 확인한다.

```bash
git diff
python scripts/validate_submission.py
python scripts/train_lgbm.py
git add .
git commit -m "add official metric and lgbm validation"
```

작업 단위는 작게 유지한다.

권장 커밋 예시:

```text
init project structure
add baseline data loader
add official metric
add time based validation
add lgbm training script
add submission validator
add wind direction features
add ensemble script
```

---

## 9. 재현성 체크리스트

최종 제출 전 반드시 확인한다.

- [ ] `train` 코드와 `inference` 코드가 분리되어 있는가?
- [ ] 최종 코드에서 원격 API를 호출하지 않는가?
- [ ] random seed가 고정되어 있는가?
- [ ] train/validation split이 시간 기준인가?
- [ ] 2025 target 값을 사용하지 않았는가?
- [ ] 예측 시점 이후 정보를 사용하지 않았는가?
- [ ] `sample_submission.csv`의 ID 컬럼을 수정하지 않았는가?
- [ ] 음수 예측값을 0으로 clip했는가?
- [ ] 용량 정보가 있다면 group capacity 기준 upper clip을 검토했는가?
- [ ] local validation score를 공식 metric으로 계산했는가?
- [ ] submission 파일 검증 스크립트를 통과했는가?
- [ ] 제출 파일명이 실험 ID와 연결되어 있는가?

---

## 10. 실험 로그 컬럼

`outputs/logs/experiments.csv`에는 최소한 아래 컬럼을 둔다.

| 컬럼 | 의미 |
|---|---|
| experiment_id | 실험 ID |
| date | 실행 날짜 |
| model | 모델 종류 |
| cv_strategy | 검증 방식 |
| train_period | 학습 기간 |
| valid_period | 검증 기간 |
| features | 주요 피처 세트 |
| total_score | 공식 metric 최종 점수 |
| one_minus_nmae | 1 - NMAE |
| ficr | FICR |
| public_score | 제출 후 public score |
| submission_path | 제출 파일 경로 |
| notes | 실험 메모 |

---

## 11. 초기 실행 순서

가장 먼저 할 일은 아래 3개다.

```text
1. open.zip 데이터 구조 확인
2. 베이스라인 재현
3. src/dacon_wind/metric.py 작성
```

이 3개가 끝나야 LightGBM, 피처 엔지니어링, 앙상블 같은 상위권용 실험을 안정적으로 시작할 수 있다.

---

## 12. 결론

수정된 최적 조합은 다음과 같다.

```text
메인 판단: GPT
메인 코딩: Codex CLI
보조 자동완성: VS Code Copilot
학습 실행: Colab
실험 기록: experiments.csv 또는 Notion
Claude Code: 사용 안 함
```

현재 단계에서는 모델을 복잡하게 만드는 것보다 **데이터 구조 파악, 베이스라인 재현, 공식 metric 구현, 시간 기반 validation 고정**이 더 중요하다.
