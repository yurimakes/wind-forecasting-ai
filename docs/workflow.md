# Workflow

## Artifact Locations

- Submission CSVs: `submissions/`
- Model artifacts: `outputs/models/`
- Prediction files: `outputs/predictions/`
- Logs and run summaries: `outputs/logs/`

## Standard Experiment Flow

1. Define an experiment id, for example `002_lgbm_time_valid`.
2. Use a time-based validation split.
3. Train with fixed seeds and save the model artifact under `outputs/models/`.
4. Save validation or inference predictions under `outputs/predictions/`.
5. Save logs, scores, and config notes under `outputs/logs/`.
6. Create a submission CSV under `submissions/` without overwriting existing submissions.
7. Validate the CSV:

```powershell
python scripts/validate_submission.py submissions/<file>.csv
```

8. Add the result to `docs/experiment_log.md`.

## Current Baseline Workflow

Updated: 2026-07-09 KST

The current strongest workflow uses baseline calendar + LDAPS/GFS mean features, 2024 time-based local validation, and separate group models for `kpx_group_1`, `kpx_group_2`, and `kpx_group_3`. Predictions are clipped by each group capacity before submission validation.

Current best submitted run:

- exp_id: `lgbm_003_tuned_submit`
- model: LightGBM tuned grid
- submission: `submissions/lgbm_003_tuned.csv`
- local validation total_score: 0.6033279875
- DACON public total_score: 0.60516
- validation command passed: `python scripts/validate_submission.py submissions/lgbm_003_tuned.csv`

CatBoost status:

- exp_id: `cat_001_baseline`
- local validation total_score: 0.5980575665
- decision: lower than `lgbm_003_tuned_submit`, so no submission yet; keep as an ensemble diversity candidate.

XGBoost status:

- exp_id: `xgb_001_baseline`
- script: `scripts/train_xgb_cv.py`
- model: XGBoost baseline
- features: baseline calendar + LDAPS/GFS mean features, no wind-derived features
- local validation total_score: 0.5988239498
- local one_minus_nmae: 0.8657252879
- local ficr: 0.3319226117
- artifacts: `outputs/predictions/xgb_001_baseline_valid_2024.csv`, `outputs/logs/xgb_001_baseline_valid_2024.json`, `outputs/models/xgb_001_baseline_valid_2024.joblib`
- decision: lower than `lgbm_003_tuned_submit`, so no submission yet; keep as an ensemble diversity candidate.

Next planned experiments:

1. `lgbm_004_weather_agg`: expanded weather aggregation features.
2. `ens_001_simple_avg`: ensemble of strong candidate submissions.

## Constraints

- Do not modify `data/raw`.
- Do not modify existing submission files unless explicitly asked.
- Do not use remote model APIs inside train or inference code.
