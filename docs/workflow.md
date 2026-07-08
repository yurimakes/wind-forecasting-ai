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

Current best public submission:

- exp_id: `ens_001_simple_avg_submit`
- model: 50/50 ensemble of `lgbm_003_tuned` and `lgbm_005_targeted_weather`
- submission: `submissions/ens_001_simple_avg.csv`
- local validation total_score: 0.6037465036
- local one_minus_nmae: 0.8680552297
- local ficr: 0.3394377775
- DACON public total_score: 0.6062263329
- DACON public one_minus_nmae: 0.8675678207
- DACON public ficr: 0.3448848451
- public rank at submission time: 275
- submitted_at: 2026-07-09 03:27:38 KST
- submission title: `ens_001_simple_avg edit`
- submitter/team display: 배추도사님
- validation command passed: `python scripts/validate_submission.py submissions/ens_001_simple_avg.csv`

Previous best public reference:

- exp_id: `lgbm_003_tuned_submit`
- DACON public total_score: 0.60516
- DACON public one_minus_nmae: 0.86678
- DACON public ficr: 0.34354

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

LightGBM weather aggregation status:

- exp_id: `lgbm_004_weather_agg`
- script: `scripts/train_lgbm_weather_agg_cv.py`
- feature function: `build_weather_agg_feature_matrix` in `src/dacon_wind/features.py`
- model: LightGBM tuned grid
- features: baseline calendar + LDAPS/GFS mean features + row-wise LDAPS/GFS weather aggregations; no wind-derived vector features
- local validation total_score: 0.5962283588
- local one_minus_nmae: 0.8657741712
- local ficr: 0.3266825463
- artifacts: `outputs/predictions/lgbm_004_weather_agg_valid_2024.csv`, `outputs/logs/lgbm_004_weather_agg_valid_2024.json`, `outputs/models/lgbm_004_weather_agg_valid_2024.joblib`
- decision: lower than `lgbm_003_tuned_submit`, so no submission.

LightGBM targeted weather status:

- exp_id: `lgbm_005_targeted_weather`
- run_id: `lgbm_005_targeted_weather_valid_2024`
- script: `scripts/train_lgbm_targeted_weather_cv.py`
- feature function: `build_targeted_weather_feature_matrix` in `src/dacon_wind/features.py`
- model: LightGBM tuned grid
- features: baseline calendar + LDAPS/GFS mean features + targeted weather features
- targeted features: near-surface and hub-height wind speeds; vertical-shear differences and ratios; GFS gust margins; LDAPS/GFS 10m wind component and speed differences
- local validation total_score: 0.6019196756
- local one_minus_nmae: 0.8674731184
- local ficr: 0.3363662327
- decision: lower than `lgbm_003_tuned`, so no standalone submission; keep as an ensemble candidate.
- interpretation: recovered most of the loss from `lgbm_004_weather_agg`; slightly improved one_minus_nmae over `lgbm_003_tuned`; lower FICR kept total_score below `lgbm_003_tuned`; useful as ensemble diversity, proven by `ens_001_simple_avg`.

Simple ensemble validation status:

- exp_id: `ens_001_simple_avg`
- validation run_id: `ens_001_simple_avg_valid_2024`
- best validation candidate: `ens_001_lgbm003_lgbm005_avg`
- validation ensemble: 0.5*`lgbm_003_tuned` + 0.5*`lgbm_005_targeted_weather`
- local validation total_score: 0.6037465036
- local one_minus_nmae: 0.8680552297
- local ficr: 0.3394377775
- decision: submission candidate.

Simple ensemble submission status:

- exp_id: `ens_001_simple_avg_submit`
- submission file: `submissions/ens_001_simple_avg.csv`
- DACON public total_score: 0.6062263329
- DACON public one_minus_nmae: 0.8675678207
- DACON public ficr: 0.3448848451
- decision: current best public submission; local validation improvement transferred to public leaderboard, improving both public 1-nMAE and public FiCR over `lgbm_003_tuned_submit`. Keep private leaderboard caution.

Experiment interpretation and lessons are maintained in `docs/experiment_feedback.md`.

Next planned experiments:

1. `ens_002_weight_search_cv`
2. `ens_003_include_xgb_selective`
3. `lgbm_006_ficr_focus`

## Constraints

- Do not modify `data/raw`.
- Do not modify existing submission files unless explicitly asked.
- Do not use remote model APIs inside train or inference code.
