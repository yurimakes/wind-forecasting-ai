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

Updated: 2026-07-10 KST

The current strongest workflow uses baseline calendar + LDAPS/GFS mean features, 2024 time-based local validation, and separate group models for `kpx_group_1`, `kpx_group_2`, and `kpx_group_3`. Predictions are clipped by each group capacity before submission validation.

Current best public submission:

- exp_id: `lgbm_008_scale_110_submit`
- model: scale-1.10 postprocessed ensemble submission
- method: scale 1.10 applied to `outputs/predictions/ens_001_simple_avg_test.csv`, clipped to group capacity
- submission: `submissions/lgbm_008_scale_110.csv`
- local validation total_score: 0.6184267418
- local one_minus_nmae: 0.8671491370
- local ficr: 0.3697043467
- DACON public total_score: 0.6211026154
- DACON public one_minus_nmae: 0.8634833072
- DACON public ficr: 0.3787219236
- public rank at submission time: 148
- submitted_at: 2026-07-10 17:54:24 KST
- submission title: `0710_v1 edit`
- validation command passed: `python scripts/validate_submission.py submissions/lgbm_008_scale_110.csv`
- caution: public gain versus `lgbm_006_ficr_focus_submit` was FiCR-driven, while public 1-nMAE dropped; final ranking depends on private leaderboard.

Previous best public reference:

- exp_id: `lgbm_006_ficr_focus_submit`
- DACON public total_score: 0.6158048399
- DACON public one_minus_nmae: 0.8679909923
- DACON public ficr: 0.3636186875
- public rank at submission time: 245

Previous single-model public reference:

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
- decision: previous best public submission before `lgbm_006_ficr_focus_submit`; local validation improvement transferred to public leaderboard, improving both public 1-nMAE and public FiCR over `lgbm_003_tuned_submit`. Keep private leaderboard caution.

FICR-focused submission status:

- exp_id: `lgbm_006_ficr_focus_submit`
- submission file: `submissions/lgbm_006_ficr_focus.csv`
- validation experiment: `lgbm_006_ficr_focus`
- selected candidate: `lgbm_006_global_scale_103`
- operation: multiply all target predictions by 1.03, then clip to group capacity
- local validation total_score: 0.6109322217
- local one_minus_nmae: 0.8692902276
- local ficr: 0.3525742158
- DACON public total_score: 0.6158048399
- DACON public one_minus_nmae: 0.8679909923
- DACON public ficr: 0.3636186875
- public rank at submission time: 245
- decision: previous best public submission before `lgbm_008_scale_110_submit`; validation-selected scaling transferred strongly to public total_score and FiCR. Keep private leaderboard caution and avoid excessive public probing.

Scale 1.10 submission status:

- exp_id: `lgbm_008_scale_110_submit`
- submission file: `submissions/lgbm_008_scale_110.csv`
- validation experiment: `lgbm_008_scale_upper_sweep_cv`
- selected scale: 1.10
- operation: multiply all target predictions by 1.10, then clip to group capacity
- local validation total_score: 0.6184267418
- local one_minus_nmae: 0.8671491370
- local ficr: 0.3697043467
- DACON public total_score: 0.6211026154
- DACON public one_minus_nmae: 0.8634833072
- DACON public ficr: 0.3787219236
- public rank at submission time: 148
- submitted_at: 2026-07-10 17:54:24 KST
- submission title: `0710_v1 edit`
- decision: current best public submission; improves public total_score by +0.0052977755 and public FiCR by +0.0151032361 versus `lgbm_006_ficr_focus_submit`, while public 1-nMAE decreased by -0.0045076851. The public gain was FiCR-driven; keep private leaderboard caution and avoid more public probing today.

Experiment interpretation and lessons are maintained in `docs/experiment_feedback.md`.

Next planned experiments:

1. Keep `lgbm_008_scale_110_submit` as the current public reference.
2. Avoid more public probing today.
3. Run validation-only modeling or feature experiments before considering any further scale submissions.

## Constraints

- Do not modify `data/raw`.
- Do not modify existing submission files unless explicitly asked.
- Do not use remote model APIs inside train or inference code.
