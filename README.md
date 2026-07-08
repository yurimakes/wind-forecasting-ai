# DACON Wind Power Forecasting

## Current Status

Updated: 2026-07-09 KST

This project is for the DACON wind power generation forecasting AI contest. The goal is to predict wind power generation for KPX groups using weather forecast data.

Main workflow:

1. Load raw train/test weather and label data.
2. Build baseline calendar + LDAPS/GFS mean features.
3. Use 2024 time-based local validation.
4. Train separate models for `kpx_group_1`, `kpx_group_2`, and `kpx_group_3`.
5. Clip predictions by each group capacity.
6. Validate every submission using `scripts/validate_submission.py`.

Raw data and generated outputs are not committed when ignored by `.gitignore`. Keep raw files under `data/raw/`, model artifacts under `outputs/models/`, prediction artifacts under `outputs/predictions/`, logs under `outputs/logs/`, and submission CSVs under `submissions/`.

## Best Result So Far

| Field | Value |
|---|---|
| exp_id | `lgbm_003_tuned_submit` |
| model | LightGBM tuned grid |
| features | baseline calendar + LDAPS/GFS mean features |
| submission file | `submissions/lgbm_003_tuned.csv` |
| local validation total_score | 0.6033279875 |
| local one_minus_nmae | 0.8673265858 |
| local ficr | 0.3393293893 |
| DACON public total_score | 0.60516 |
| DACON public one_minus_nmae | 0.86678 |
| DACON public ficr | 0.34354 |
| public rank at submission time | 277 |
| submitted name | 배추 |
| submitter | 배추도사님 |
| submitted_at_kst | 2026-07-09 |

Generated artifacts from the final tuned submission run:

- `submissions/lgbm_003_tuned.csv`
- `outputs/predictions/lgbm_003_tuned_test.csv`
- `outputs/logs/lgbm_003_tuned_submit.json`
- `outputs/models/lgbm_003_tuned_submit.joblib`

Validation status:

- `python scripts/train_lgbm_tuned_submit.py` completed successfully.
- `python scripts/validate_submission.py submissions/lgbm_003_tuned.csv` passed.

## Experiment Summary

| exp_id | model | features | local total_score | public total_score | status |
|---|---|---|---:|---:|---|
| `baseline_rf_001` | RandomForest | baseline features | - | 0.5879246832 | submitted first baseline |
| `lgbm_001` | LightGBM v1 | baseline calendar + LDAPS/GFS mean features | 0.5984976879 | - | local validation baseline |
| `lgbm_002_wind` | LightGBM v1 | baseline calendar + LDAPS/GFS mean features + wind vector derivatives | 0.5966447814 | - | worse than `lgbm_001`, no submission |
| `lgbm_003_tuned_submit` | LightGBM tuned grid | baseline calendar + LDAPS/GFS mean features | 0.6033279875 | 0.60516 | current best submitted model |
| `cat_001_baseline` | CatBoost baseline | baseline calendar + LDAPS/GFS mean features | 0.5980575665 | - | ensemble candidate, no submission yet |
| `xgb_001_baseline` | XGBoost baseline | baseline calendar + LDAPS/GFS mean features | 0.5988239498 | - | ensemble candidate, no submission yet |

Important lessons so far:

- Wind vector derivative features in `lgbm_002_wind` did not improve local validation.
- Tuned LightGBM with smaller trees performed better.
- Current best LightGBM setting: `num_leaves=15`, `min_child_samples=20`, `learning_rate=0.03`, `n_estimators=1000`, `reg_lambda=5.0`.
- CatBoost baseline did not beat tuned LightGBM but may be useful later for ensemble diversity.
- XGBoost baseline did not beat tuned LightGBM but is currently the strongest non-LightGBM ensemble candidate.

## How to Reproduce Current Best Submission

Run from the project root:

```powershell
python scripts/tune_lgbm_cv.py
python scripts/train_lgbm_tuned_submit.py
python scripts/validate_submission.py submissions/lgbm_003_tuned.csv
```

The tuned submission uses separate LightGBM models for the three KPX groups, baseline calendar + LDAPS/GFS mean features, and group-capacity clipping.

## Next Experiments

`lgbm_003_tuned_submit` is the current best baseline for comparison. CatBoost and XGBoost are currently ensemble candidates, not standalone submission candidates. The next priority is to create stronger features and ensemble candidates rather than only micro-tuning LightGBM.

Planned experiments:

1. `lgbm_004_weather_agg`: expanded weather aggregation features.
2. `ens_001_simple_avg`: ensemble of strong candidate submissions.
