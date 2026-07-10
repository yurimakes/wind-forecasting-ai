# Wind Power Forecasting

## Current Status

Updated: 2026-07-10 KST

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
| exp_id | `lgbm_008_scale_110_submit` |
| model | scale-1.10 postprocessed ensemble submission |
| features | scale 1.10 applied to `ens_001_simple_avg_test` predictions, capacity clipped |
| submission file | `submissions/lgbm_008_scale_110.csv` |
| local validation total_score | 0.6184267418 |
| local one_minus_nmae | 0.8671491370 |
| local ficr | 0.3697043467 |
| DACON public total_score | 0.6211026154 |
| DACON public one_minus_nmae | 0.8634833072 |
| DACON public ficr | 0.3787219236 |
| public rank at submission time | 148 |
| submission title | `0710_v1 edit` |
| submitted_at_kst | 2026-07-10 17:54:24 |

Previous best public references:

- `ens_001_simple_avg_submit`: public total_score=0.6062263329, public one_minus_nmae=0.8675678207, public ficr=0.3448848451, rank 275 at submission time.
- `lgbm_003_tuned_submit`: public total_score=0.60516, public one_minus_nmae=0.86678, public ficr=0.34354.

Previous best public reference:

- `lgbm_006_ficr_focus_submit`: public total_score=0.6158048399, public one_minus_nmae=0.8679909923, public ficr=0.3636186875, rank 245 at submission time.

Generated artifacts from the current best public submission run:

- `submissions/lgbm_008_scale_110.csv`
- `outputs/predictions/lgbm_008_scale_110_test.csv`
- `outputs/logs/lgbm_008_scale_110_submit.json`

Validation status:

- `python scripts/validate_submission.py submissions/lgbm_008_scale_110.csv` passed.

## Experiment Summary

| exp_id | model | features | local total_score | public total_score | status |
|---|---|---|---:|---:|---|
| `baseline_rf_001` | RandomForest | baseline features | - | 0.5879246832 | submitted first baseline |
| `lgbm_001` | LightGBM v1 | baseline calendar + LDAPS/GFS mean features | 0.5984976879 | - | local validation baseline |
| `lgbm_002_wind` | LightGBM v1 | baseline calendar + LDAPS/GFS mean features + wind vector derivatives | 0.5966447814 | - | worse than `lgbm_001`, no submission |
| `lgbm_003_tuned_submit` | LightGBM tuned grid | baseline calendar + LDAPS/GFS mean features | 0.6033279875 | 0.60516 | previous best submitted model and comparison baseline |
| `lgbm_004_weather_agg` | LightGBM tuned grid | baseline calendar + LDAPS/GFS mean features + row-wise LDAPS/GFS weather aggregations | 0.5962283588 | - | worse than `lgbm_003_tuned`, no submission |
| `lgbm_005_targeted_weather` | LightGBM tuned grid | baseline calendar + LDAPS/GFS mean features + targeted weather features | 0.6019196756 | - | ensemble candidate, not standalone submission |
| `ens_001_simple_avg` | validation-only simple ensemble | 0.5*`lgbm_003_tuned` + 0.5*`lgbm_005_targeted_weather` | 0.6037465036 | - | selected submission candidate |
| `ens_001_simple_avg_submit` | 50/50 ensemble submission | 0.5*`lgbm_003_tuned` + 0.5*`lgbm_005_targeted_weather` | 0.6037465036 | 0.6062263329 | previous best public submission |
| `lgbm_006_ficr_focus_submit` | FICR-focused postprocessed ensemble submission | 1.03 global scaling of `ens_001_simple_avg_test` predictions, capacity clipped | 0.6109322217 | 0.6158048399 | previous best public submission |
| `lgbm_008_scale_110_submit` | scale-1.10 postprocessed ensemble submission | 1.10 global scaling of `ens_001_simple_avg_test` predictions, capacity clipped | 0.6184267418 | 0.6211026154 | current best public submission |
| `cat_001_baseline` | CatBoost baseline | baseline calendar + LDAPS/GFS mean features | 0.5980575665 | - | ensemble candidate, no submission yet |
| `xgb_001_baseline` | XGBoost baseline | baseline calendar + LDAPS/GFS mean features | 0.5988239498 | - | ensemble candidate, no submission yet |

Important lessons so far:

- Wind vector derivative features in `lgbm_002_wind` did not improve local validation.
- Broad row-wise LDAPS/GFS aggregation in `lgbm_004_weather_agg` degraded performance, likely because it mixed weather variables with different physical meanings and units.
- Targeted weather features in `lgbm_005_targeted_weather` recovered most of the `lgbm_004_weather_agg` loss and slightly improved one_minus_nmae over `lgbm_003_tuned`, but lower FICR kept the standalone total_score below `lgbm_003_tuned`.
- `lgbm_005_targeted_weather` is useful ensemble diversity: the 50/50 `ens_001_simple_avg` validation improvement transferred to the DACON public leaderboard.
- `lgbm_008_scale_110_submit` is the current best public submission; validation-selected 1.10 global scaling of `ens_001_simple_avg_test` predictions improved public total_score mainly through FiCR, while public one_minus_nmae dropped versus `lgbm_006_ficr_focus_submit`.
- Tuned LightGBM with smaller trees performed better.
- Current best LightGBM setting: `num_leaves=15`, `min_child_samples=20`, `learning_rate=0.03`, `n_estimators=1000`, `reg_lambda=5.0`.
- CatBoost baseline did not beat tuned LightGBM but may be useful later for ensemble diversity.
- XGBoost baseline did not beat tuned LightGBM but is currently the strongest non-LightGBM ensemble candidate.
- Detailed qualitative feedback and experiment lessons are documented in `docs/experiment_feedback.md`.

## How to Reproduce Previous Tuned LightGBM Submission

Run from the project root:

```powershell
python scripts/tune_lgbm_cv.py
python scripts/train_lgbm_tuned_submit.py
python scripts/validate_submission.py submissions/lgbm_003_tuned.csv
```

The tuned submission uses separate LightGBM models for the three KPX groups, baseline calendar + LDAPS/GFS mean features, and group-capacity clipping.

## Next Experiments

`lgbm_008_scale_110_submit` is the current best public submission, with public total_score=0.6211026154, public one_minus_nmae=0.8634833072, and public ficr=0.3787219236. It uses scale 1.10 applied to `ens_001_simple_avg_test` predictions, clipped to group capacity. The public gain versus `lgbm_006_ficr_focus_submit` is FiCR-driven, while public one_minus_nmae decreased. `ens_001_simple_avg_submit` remains the pre-scaling ensemble reference, and `lgbm_003_tuned_submit` remains the main single-model comparison baseline. CatBoost and XGBoost are currently ensemble candidates, not standalone submission candidates. Detailed qualitative feedback and experiment lessons are documented in `docs/experiment_feedback.md`.

Planned experiments:

1. Keep `lgbm_008_scale_110_submit` as the current public reference.
2. Avoid more public probing today.
3. Run validation-only modeling or feature experiments before considering any further scale submissions.
