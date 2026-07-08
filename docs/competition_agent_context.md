# 제3회 풍력발전량 예측 AI 경진대회 - 프로젝트/에이전트 컨텍스트

> 권장 저장 위치: `docs/competition_agent_context.md`  
> 목적: ChatGPT, Codex CLI, VS Code Copilot 등 AI 개발 보조 도구가 이 대회의 규칙·평가식·금지사항·프로젝트 방향을 일관되게 이해하도록 하는 문서

---

## 0. 소스 우선순위

이 문서는 다음 두 자료를 합쳐 정리한 작업용 컨텍스트다.

1. **데이콘 공식 대회 페이지/규칙 문서**: `데이콘 풍력발전량예측 대회규칙설명.docx`
2. **사전 설명회 유튜브 스크립트**: `붙여넣은 텍스트 (1).txt`

우선순위는 다음과 같다.

1. 데이콘 공식 대회 페이지, 데이터 탭, 평가 코드, FAQ, 공지사항
2. 공식 대회규칙 설명 DOCX
3. 사전 설명회 스크립트
4. 이 프로젝트의 내부 가정/실험 메모

서로 충돌하는 내용이 있으면 **공식 대회 페이지와 공식 평가 코드**를 최우선으로 따른다.

자동 자막에는 오타가 많다. 예: `폭력`은 `풍력`, `정상금`은 `정산금`, `출론`은 `추론`, `CSB`는 `CSV`로 해석한다. `KPX/KFX`, `LDAPS/애답스/에더스`, `GFS/JFS`처럼 자막이 불확실한 용어는 반드시 공식 데이터 명세의 컬럼명과 파일명을 우선한다.

---

## 1. 대회 핵심 요약

- 대회명: **제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026**
- 주제: **기상예보 데이터 기반 풍력발전량 예측 AI 모델 개발**
- 문제 유형: 정형 데이터 기반 회귀 예측
- 예측 대상: 특정 풍력단지의 **3개 그룹별 시간 단위 발전량**
- 제출 대상: 테스트 기간의 각 시간별 `kpx_group_1`, `kpx_group_2`, `kpx_group_3` 발전량 예측값
- 핵심 평가 관점:
  - 예측 오차가 작은가?
  - 재생에너지 발전량 예측제도 관점에서 정산금을 많이 확보할 수 있는가?

이 대회는 단순히 NMAE만 낮추는 대회가 아니다. 최종 점수는 **1-NMAE**와 **정산금획득률(FICR)**을 50:50으로 반영한다.

---

## 2. 데이터 이해

### 2.1 제공 데이터 개념

설명회 기준으로 데이터는 크게 다음 성격을 가진다.

- 기상예보 데이터
  - 학습 구간: 2022년 ~ 2024년
  - 테스트 구간 입력용으로도 제공
  - 원천 대용량 자료를 그대로 주는 것이 아니라, 풍력발전기 주변 영역 중심으로 가공된 형태로 제공되는 것으로 설명됨
- SCADA/운영 실측 데이터
  - 실제 풍력발전기 운영 및 계측 실측 데이터 성격
  - **학습 데이터에서만 제공**
  - 그룹 1, 2는 2022년부터 존재
  - 그룹 3은 2023년부터 존재한다고 설명됨

공식 데이터 탭이 공개되면 파일명, 컬럼명, 시간대, 결측 구조, forecast issue time/base time, target time을 다시 확인해야 한다.

### 2.2 제출 형태

제출 파일은 공식 `sample_submission.csv`의 컬럼명과 순서를 반드시 따른다. 설명회에서는 2025년 전체에 대해 시간 단위로 3개 그룹의 발전량을 예측한다고 설명했다.

예상 형태는 다음과 비슷하되, 실제 구현은 반드시 sample submission을 기준으로 한다.

```csv
forecast_id,forecast_kst_dtm,kpx_group_1,kpx_group_2,kpx_group_3
...
```

### 2.3 외부 데이터

외부 데이터는 사용할 수 있다. 예를 들어 공개 지형 데이터, 공개 기상 데이터, 공개 지리 정보 등은 후보가 될 수 있다. 다만 아래 조건을 모두 만족해야 한다.

- 누구나 접근 가능한 공개 데이터
- 법적 제약이 없는 데이터
- 저작권, 라이선스, 개인정보, 이용약관 위반이 없는 데이터
- 예측기준시점 기준 Data Leakage가 없는 데이터
- 2차 평가 대상이 되었을 때 출처, 수집 방법, 수집 시점, 사용 변수, 라이선스, 전처리 코드, 재현 방법을 소명할 수 있는 데이터

외부 데이터를 쓰는 경우 반드시 `docs/external_data_registry.md` 또는 동등한 문서에 기록한다.

---

## 3. 평가식

### 3.1 총점

공식 평가 산식:

```text
Score = 0.5 * (1 - NMAE) + 0.5 * FICR
```

단, 여기서 NMAE와 FICR은 3개 그룹별 값을 계산한 뒤 평균하는 구조다.

### 3.2 1-NMAE

그룹별 NMAE:

```text
group_NMAE = mean(abs(pred - actual) / group_capacity)
```

전체 1-NMAE:

```text
1-NMAE = 1 - mean(group_NMAE for 3 groups)
```

값이 높을수록 좋다.

### 3.3 FICR: 정산금획득률

그룹별 FICR:

```text
group_FICR = earned_settlement / theoretical_max_settlement
FICR = mean(group_FICR for 3 groups)
```

설명회 기준 시간별 정산금은 시간별 normalized error 기준으로 다음처럼 설명됐다.

```text
hourly_normalized_error <= 0.06  -> 정산금 점수 4
hourly_normalized_error <= 0.08  -> 정산금 점수 3
otherwise                       -> 0
```

공식 평가 코드가 공개되어 있으면 구현은 반드시 공식 코드와 일치시킨다.

### 3.4 평가 제외 조건

평가는 **실제 발전량이 해당 그룹 설비용량의 10% 이상인 시간대만** 대상으로 한다.

주의: 테스트 시점에는 실제 발전량을 알 수 없으므로, 이 조건을 직접 이용해 테스트 행을 제거하거나 보정하면 안 된다. 다만 학습/검증에서는 이 특성을 고려해 validation metric, 샘플 가중치, 저발전 구간 분석 전략을 설계할 수 있다.

### 3.5 현재 프로젝트 metric 구현 관련 메모

이 프로젝트에서 metric 구현을 할 때는 다음을 반드시 확인한다.

- 공식 평가 코드와 같은 capacity 값을 사용한다.
- 현재 프로젝트에서 사용 중인 것으로 보이는 상수는 다음과 같으나, 공식 평가 코드로 재확인해야 한다.

```python
CAPACITY = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}
```

- `metric.py`는 최소한 다음 값을 계산/반환할 수 있어야 한다.
  - `total_score`
  - `one_minus_nmae`
  - `ficr`
  - group별 NMAE/FICR 디테일
- `y_true`, `y_pred`는 pandas DataFrame 입력을 지원하게 둔다.
- validation에서는 공식 평가 제외 조건, FICR threshold를 그대로 반영한다.

---

## 4. Data Leakage 방지 규칙

이 대회에서 가장 중요한 규칙이다. 모든 코드 작성과 피처 엔지니어링은 아래 원칙을 반드시 지킨다.

### 4.1 예측기준시점

- 예측기준시점은 **예측일 전날 14:00 KST**다.
- 한국 시간 기준이다.
- 외부 데이터가 UTC 등 다른 시간대로 제공되면 반드시 KST로 변환해 비교한다.
- 데이터 사용 가능 여부는 데이터가 가리키는 대상 시각이 아니라, 그 데이터가 **생성·공개·확정되어 실제로 활용 가능해진 시각**을 기준으로 판단한다.

예:

```text
2025-01-13 10:00 발전량을 예측하는 경우
예측기준시점 = 2025-01-12 14:00 KST
```

이때 2025-01-13 00:00에 생성/공개된 2025-01-13 10:00 대상 예보자료는 사용할 수 없다. 예측기준시점보다 나중에 공개된 정보이기 때문이다.

### 4.2 금지되는 정보

아래 정보는 사용하면 안 된다.

- 평가 구간의 실제 발전량
- 평가 구간의 실측 운영 데이터
- 비공개 운영 데이터
- 정답 또는 정답에 준하는 정보
- 예측기준시점 이후 생성/공개/확정된 관측값, 실측값, 사후 보정자료, 재분석자료
- 테스트 정답을 유추할 수 있는 외부 데이터
- 평가 데이터에 대한 pseudo-labeling 또는 정답성 정보 활용

### 4.3 파생변수에도 동일 적용

Data Leakage 금지는 원본 데이터뿐 아니라 파생변수에도 적용된다.

금지 예시:

- target time 이후의 실측값을 포함한 rolling mean
- 2025년 전체 정보를 한 번에 집계한 뒤 테스트 행에 붙이는 통계값
- 테스트 기간 전체 분포를 보고 만든 보정값
- 예측기준시점 이후 공개된 예보를 target time에 맞춰 붙이는 것
- 사후 재분석자료를 실제 예보처럼 사용하는 것

허용 방향:

- 각 예측 행마다 `available_at <= prediction_cutoff` 조건을 만족하는 데이터만 merge
- lag/rolling feature는 반드시 과거 방향으로만 생성
- 외부 데이터는 source issue time, publish time, download time, license를 함께 기록

### 4.4 코드 레벨 권장 규칙

모든 데이터 merge 함수는 가능하면 아래 컬럼 개념을 명시한다.

```text
target_time_kst      # 예측 대상 시각
issue_time_kst       # 예보/자료 생성 시각
published_time_kst   # 실제 공개/활용 가능해진 시각, 알 수 없으면 issue_time 기준으로 보수적으로 처리
prediction_cutoff_kst# 예측기준시점: target date 전날 14:00 KST
```

외부 데이터 join 시에는 최소한 다음 검사를 넣는다.

```python
assert used_data_time_kst <= prediction_cutoff_kst
```

공식 FAQ에서 기준시점 포함 여부가 명확해지기 전까지는 보수적으로 `used_data_time_kst < prediction_cutoff_kst`에 가깝게 처리하는 것을 권장한다.

---

## 5. 모델/도구 사용 규칙

### 5.1 허용 언어

- Python

### 5.2 사전학습 모델

- 2026년 7월 5일까지 공식적으로 가중치가 공개된 오픈소스 모델만 사용 가능
- 단순히 가중치가 공개되어 있어도 비상업적/연구용/평가용 등 제한 라이선스면 사용 불가
- 사용 모델, 가중치, 라이선스, 다운로드 경로를 산출물 검증 시 설명할 수 있어야 함

### 5.3 API 기반 모델 사용 제한

추론에 원격 API 기반 모델을 사용하면 안 된다.

금지 예:

- OpenAI API
- Gemini API
- Claude API
- Hugging Face Inference API
- Together AI
- OpenRouter
- 기타 원격 서버에서 모델 응답을 받아 예측값을 만드는 방식

AI 도구 사용 원칙:

- ChatGPT, Codex CLI, Copilot 등은 **개발 보조**로만 사용한다.
- 최종 `train`/`inference` 코드 내부에서 원격 API 모델을 호출하지 않는다.
- 모델 추론은 참가자가 직접 관리하는 PC/클라우드/서버에서 모델 가중치를 직접 로드하는 방식이어야 한다.

---

## 6. 제출/리더보드/최종 산출물

### 6.1 제출 제한

- 1일 최대 제출 횟수: 5회
- 개인/팀 모두 하나의 참가 단위로 간주한다.
- 팀이라고 해서 인원수만큼 제출 횟수가 늘어나지 않는다.
- 제출 CSV는 UTF-8 인코딩을 적용한다.

### 6.2 Public / Private 리더보드

- Public Score: 전체 평가 데이터 중 사전 샘플링된 40%
- Private Score: 나머지 60%
- 최종 1차 평가는 Private Score 기준
- Public 점수는 모델 개선용 피드백일 뿐 최종 수상 기준이 아니다.
- Public에 과적합하지 않도록 validation split과 모델 선택 기준을 별도로 유지한다.

### 6.3 최종 채점 파일 선택

- 제출할 때마다 순위가 가장 높은 파일이 자동 선택될 수 있다.
- 최종으로 채점받고 싶은 파일이 Public 최고점 파일이 아니라면 제출 탭에서 직접 선택해야 한다.
- 새 제출을 하면 자동 선택 상태가 다시 바뀔 수 있으므로 대회 종료 전 반드시 확인한다.

### 6.4 2차 평가 대상 산출물

Private 리더보드 상위 30팀(예비 10팀 포함)은 산출물을 제출해야 한다. 산출물 검증을 통과한 상위 20팀이 오프라인 발표평가 대상이 된다.

필수 제출물:

- Private Score 복원이 가능한 코드와 모델 파일
- 학습 코드와 추론 코드 분리
  - `train.py` 또는 train notebook
  - `inference.py` 또는 inference notebook
- 외부 데이터 사용 시 사용한 모든 외부 데이터 파일
- 외부 데이터 출처, 수집 방법, 수집 시점, 사용 기간, 사용 변수, 라이선스, 전처리 코드
- 개발 환경(OS) 및 라이브러리 버전
- UTF-8 인코딩 코드/주석
- 오류 없이 실행 가능한 코드
- 발표 10분 분량 PDF
  - 기술적 오류 방지를 위해 발표자료는 PPT가 아니라 PDF 제출
- 참가 자격 증빙 서류

메일 제목 형식:

```text
[팀명] 제3회 풍력발전량 예측 AI 경진대회 산출물 제출
```

---

## 7. 일정

- 참가 신청 시작: 2026-06-12
- 사전 워크샵: 2026-06-26
- 대회 시작: 2026-07-06 10:00
- 팀 병합 마감: 2026-08-07
- 대회 종료: 2026-08-14 09:59 / 제출 마감 안내상 10:00 기준 확인 필요
- 2차 평가 대상자 산출물 제출 마감: 2026-08-17 10:00
- 산출물 검증 종료: 2026-08-21
- 오프라인 발표 평가: 2026-08-28 예정
- 시상 및 컨퍼런스: 2026-09-04

일정은 대회 중 변경될 수 있으므로 데이콘 공지사항을 주기적으로 확인한다.

---

## 8. 프로젝트 개발 전략

### 8.1 기본 방향

1. 공식 baseline으로 전체 제출 사이클을 먼저 성공시킨다.
2. 공식 metric을 로컬에서 재현한다.
3. 시간 기반 validation을 만든다.
4. leakage-safe feature pipeline을 만든다.
5. LightGBM/XGBoost/CatBoost 중심의 tabular regression baseline을 강화한다.
6. Public LB에 과적합하지 말고 Private robustness를 우선한다.
7. 산출물 검증을 대비해 실험 로그와 외부 데이터 기록을 대회 초반부터 남긴다.

### 8.2 Validation 설계

권장 validation:

- 2022~2023 학습, 2024 검증
- 또는 월/계절을 고려한 time-series split
- 그룹별 점수, 계절별 점수, 풍속 구간별 점수, 저발전/고발전 구간별 점수를 따로 본다.

주의:

- 랜덤 split은 시계열 누수 가능성이 크므로 기본 검증으로 쓰지 않는다.
- 같은 target time 주변의 예보/실측이 train/valid에 섞이지 않도록 한다.
- validation metric은 official metric과 최대한 동일하게 구현한다.

### 8.3 Feature Engineering 후보

기상 예보 기반:

- 풍속, 풍향, 기온, 습도, 기압 등 기본 변수
- 풍향의 sin/cos 변환
- 풍속 제곱/세제곱 계열 변수
- hub-height 보정 가능성 검토
- 예보 lead time
- forecast issue time, target time, lead hour
- grid 주변 평균/최대/최소/표준편차
- 발전단지/그룹별 공간 집계

시간 기반:

- hour, dayofweek, month, season
- 일출/일몰 또는 태양고도 후보는 풍력에서 우선순위 낮지만 계절성 보정으로 검토 가능
- 공휴일 효과는 직접 발전량과 관련성이 낮아 우선순위 낮음

SCADA/발전량 기반:

- 학습 구간에서는 과거 발전량 lag/rolling 가능
- 단, 테스트 추론 시 같은 방식으로 실제 사용 가능한 과거 정보만 있어야 한다.
- 예측기준시점 이후의 실제 발전량 lag를 사용하면 leakage다.

외부 데이터 후보:

- 공개 지형/고도/경사/방위 데이터
- 공개 지리 좌표 기반 거리/방향 변수
- 공개 기상 관측 데이터는 예측기준시점 이전 공개 여부를 반드시 확인해야 하므로 주의
- 재분석자료는 사후 확정 자료일 가능성이 높아 원칙적으로 위험하다.

### 8.4 Post-processing 후보

- 예측값은 물리적으로 가능한 범위로 clip한다.

```python
pred = pred.clip(lower=0, upper=group_capacity)
```

- 그룹별 설비용량 초과 예측 방지
- 음수 예측 방지
- FICR threshold를 고려해 6%, 8% 경계 부근의 calibration 실험
- Public 점수만 보고 과도한 보정 금지

---

## 9. 권장 프로젝트 구조

```text
wind-forecasting-ai-contest/
├─ README.md
├─ AGENTS.md
├─ docs/
│  ├─ competition_agent_context.md
│  ├─ experiment_log.md
│  ├─ external_data_registry.md
│  └─ submission_log.md
├─ data/
│  ├─ raw/              # 원본 데이터. git 추적 제외
│  ├─ interim/          # 중간 산출 데이터. git 추적 제외
│  ├─ processed/        # 학습용 가공 데이터. git 추적 제외
│  └─ external/         # 외부 데이터. 출처 문서 필수
├─ notebooks/
│  ├─ Baseline.ipynb
│  ├─ EDA.ipynb
│  └─ Validation.ipynb
├─ src/
│  └─ dacon_wind/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ data.py
│     ├─ features.py
│     ├─ leakage.py
│     ├─ metric.py
│     ├─ train.py
│     └─ inference.py
├─ models/              # 모델 파일. 필요 시 git 추적 제외
├─ outputs/
│  ├─ submissions/
│  └─ reports/
└─ requirements.txt
```

---

## 10. Codex/AI 에이전트 작업 규칙

Codex CLI 또는 다른 AI 에이전트에게 이 프로젝트 작업을 맡길 때는 아래 규칙을 반드시 지키게 한다.

### 10.1 절대 금지

- `train.py` 또는 `inference.py`에서 OpenAI/Gemini/Claude/HF Inference API 등 원격 모델 API 호출 금지
- 평가 데이터의 실제 발전량 또는 정답성 정보 활용 금지
- 예측기준시점 이후 공개된 외부 데이터 사용 금지
- 테스트 기간 전체 통계를 이용한 feature 생성 금지
- sample submission 형식 임의 변경 금지
- data/output/model 대용량 파일 git commit 금지

### 10.2 필수 구현 방향

- 공식 metric 구현 및 단위 테스트
- train/inference 코드 분리
- 실험별 config 저장
- submission 파일 생성 함수 분리
- 외부 데이터 사용 시 registry 문서 갱신
- 모든 시간 컬럼은 KST/UTC 여부를 명확히 표시
- leakage check 유틸리티 구현

### 10.3 Codex에게 줄 수 있는 기본 프롬프트

```text
Read docs/competition_agent_context.md first. Follow the DACON BARAM 2026 rules strictly.
Do not use remote API-based models in train or inference.
Do not introduce data leakage: every feature must be available before the prediction cutoff, which is the day before target date at 14:00 KST.
Keep train and inference code separated.
Implement and test the official metric before optimizing models.
Do not modify raw data or submission files unless explicitly requested.
```

---

## 11. 산출물 검증 대비 체크리스트

대회 중 계속 관리한다.

- [ ] 공식 데이터 파일명/컬럼명 정리
- [ ] sample submission 컬럼/순서 확인
- [ ] 공식 metric 코드와 `src/dacon_wind/metric.py` 일치 확인
- [ ] train/inference 분리 완료
- [ ] `requirements.txt` 또는 `environment.yml` 정리
- [ ] OS/Python/library version 기록
- [ ] random seed 고정
- [ ] 모델 파일 저장 경로 정리
- [ ] 외부 데이터 사용 여부 결정
- [ ] 외부 데이터 registry 작성
- [ ] leakage-safe merge 검증
- [ ] 실험 로그 작성
- [ ] 제출 로그 작성
- [ ] 최종 선택 제출 파일 확인
- [ ] 발표 PDF 초안 대회 중 병행 작성

---

## 12. 외부 데이터 registry 템플릿

외부 데이터를 사용하면 아래 형식으로 기록한다.

```markdown
# External Data Registry

| ID | 데이터명 | 출처 URL | 라이선스/약관 | 수집일 | 데이터 생성/공개 시각 기준 | 사용 기간 | 사용 변수 | 전처리 파일 | Leakage 검토 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|
| ext_001 | 예: 공개 고도 데이터 |  |  | 2026-07-xx | 사전 고정 지형 데이터 | 전체 | elevation | scripts/preprocess_x.py | 시간 누수 없음 |  |
```

---

## 13. 현재 열려 있는 확인 사항

공식 데이터 다운로드 후 반드시 확인해야 할 것:

1. 기상예보 데이터의 정확한 이름: LDAPS/GFS 여부
2. 각 파일의 시간 컬럼 의미: issue/base/forecast/target time
3. 시간대: KST/UTC 여부
4. 그룹별 설비용량 공식 값
5. sample submission 컬럼명과 행 수
6. 공식 metric 코드의 FICR threshold와 clipping/minmax 처리 방식
7. 예측기준시점 포함 여부: `<= cutoff`인지 `< cutoff`인지 FAQ 확인
8. 외부 데이터 허용 범위에 대한 FAQ 업데이트

---

## 14. 최종 전략 메모

- Public 점수 1등보다 Private에서 안정적인 모델이 중요하다.
- `1-NMAE`와 `FICR`이 모두 리더보드에 표시되므로 두 지표를 따로 추적한다.
- FICR은 threshold 기반이라 작은 보정이 점수에 크게 작용할 수 있다.
- 단, FICR만 보고 과도하게 보정하면 NMAE가 나빠질 수 있으므로 총점 기준으로 판단한다.
- 대회 종료 후 산출물 제출 기간이 짧으므로, 코드는 처음부터 재현 가능하게 관리한다.
- 발표 평가는 문제 이해, 데이터 구성/분석, 성능 개선 과정, 인사이트와 성능 향상의 연결 논리를 본다.
- 실험 과정에서 실패한 시도도 발표 자료에서 문제 해결력으로 설명할 수 있게 기록한다.

---

## Current Progress Snapshot - 2026-07-09 KST

- Competition: DACON wind power generation forecasting AI contest.
- Project goal: predict wind power generation for KPX groups using weather forecast data.
- Current workflow: load raw train/test weather and label data, build baseline calendar + LDAPS/GFS mean features, use 2024 time-based local validation, train separate models for `kpx_group_1`, `kpx_group_2`, and `kpx_group_3`, clip predictions by group capacity, and validate every submission with `scripts/validate_submission.py`.
- Current best submitted model: `lgbm_003_tuned_submit`.
- Best submission file: `submissions/lgbm_003_tuned.csv`.
- Best local validation metrics: total_score=0.6033279875, one_minus_nmae=0.8673265858, ficr=0.3393293893.
- Best DACON public metrics: total_score=0.60516, one_minus_nmae=0.86678, ficr=0.34354.
- Public rank at submission time: 277.
- Submitted name: 배추.
- Submitter: 배추도사님.
- Final tuned run artifacts: `submissions/lgbm_003_tuned.csv`, `outputs/predictions/lgbm_003_tuned_test.csv`, `outputs/logs/lgbm_003_tuned_submit.json`, `outputs/models/lgbm_003_tuned_submit.joblib`.
- Validation status: `python scripts/train_lgbm_tuned_submit.py` completed successfully, and `python scripts/validate_submission.py submissions/lgbm_003_tuned.csv` passed.
- CatBoost experiment: `cat_001_baseline` used the same baseline feature matrix with no wind-derived features and scored local total_score=0.5980575665, one_minus_nmae=0.8672565037, ficr=0.3288586294. It is lower than `lgbm_003_tuned_submit`, so it is an ensemble candidate rather than a standalone submission candidate.
- XGBoost experiment: `xgb_001_baseline` used `scripts/train_xgb_cv.py` with the same baseline feature matrix and no wind-derived features. It scored local total_score=0.5988239498, one_minus_nmae=0.8657252879, ficr=0.3319226117. Artifacts are `outputs/predictions/xgb_001_baseline_valid_2024.csv`, `outputs/logs/xgb_001_baseline_valid_2024.json`, and `outputs/models/xgb_001_baseline_valid_2024.joblib`. It is lower than `lgbm_003_tuned_submit`, so no submission yet; keep it as an ensemble candidate.
- LightGBM weather aggregation experiment: `lgbm_004_weather_agg` used `scripts/train_lgbm_weather_agg_cv.py` and `build_weather_agg_feature_matrix` with baseline calendar + LDAPS/GFS mean features plus row-wise LDAPS/GFS weather aggregations, with no wind-derived vector features. It scored local total_score=0.5962283588, one_minus_nmae=0.8657741712, ficr=0.3266825463. Artifacts are `outputs/predictions/lgbm_004_weather_agg_valid_2024.csv`, `outputs/logs/lgbm_004_weather_agg_valid_2024.json`, and `outputs/models/lgbm_004_weather_agg_valid_2024.joblib`. It did not beat `lgbm_003_tuned` local total_score=0.6033279875, so no submission. Detailed interpretation is in `docs/experiment_feedback.md`.
- Current direction: keep `lgbm_003_tuned_submit` as the comparison baseline, prioritize `lgbm_005_targeted_weather` next, then test `ens_001_simple_avg`.
