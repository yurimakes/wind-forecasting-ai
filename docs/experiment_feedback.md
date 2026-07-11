# Experiment Feedback

Updated: 2026-07-11 KST

This file records qualitative feedback, experiment interpretation, failure analysis, and next actions for the DACON wind power generation forecasting project.

Use this file together with `docs/experiment_log.md`:

- `docs/experiment_log.md`: numeric experiment/submission history
- `docs/experiment_feedback.md`: interpretation, lessons, and next experiment decisions

---

## Current Reference Baseline

### `lgbm_008_scale_110_submit`

Current best public submission.

| Item | Value |
|---|---:|
| local total_score | 0.6184267418 |
| local one_minus_nmae | 0.8671491370 |
| local ficr | 0.3697043467 |
| DACON public total_score | 0.6211026154 |
| DACON public one_minus_nmae | 0.8634833072 |
| DACON public ficr | 0.3787219236 |
| public rank at submission time | 148 |

Key interpretation:

- The 1.10 globally scaled and capacity-clipped `ens_001_simple_avg_test` prediction is the current public reference.
- Public total_score improved by +0.0052977755 versus `lgbm_006_ficr_focus_submit`.
- Public FiCR improved by +0.0151032361 versus `lgbm_006_ficr_focus_submit`.
- Public 1-NMAE decreased by -0.0045076851 versus `lgbm_006_ficr_focus_submit`.
- The public score increase is mainly FiCR-driven.
- Keep caution that the final target is the private leaderboard, not public score alone.
- Avoid more public probing today because repeated public feedback can overfit model selection to the public split.
- `lgbm_006_ficr_focus_submit` remains the previous best public reference.
- `ens_001_simple_avg_submit` remains the pre-scaling ensemble reference.
- `lgbm_003_tuned_submit` remains the main single-model comparison baseline.
- `ens_004_group_scale_submit` underperformed this reference on public score and is not current best.
- The best LightGBM setting used a relatively small tree structure:
  - `num_leaves=15`
  - `min_child_samples=20`
  - `learning_rate=0.03`
  - `n_estimators=1000`
  - `reg_lambda=5.0`
- Smaller, more regularized trees appear better than larger, more complex trees for the current feature set.
- Use `lgbm_008_scale_110_submit` as the public reference, `lgbm_006_ficr_focus_submit` as the previous scaled public reference, and `lgbm_003_tuned_submit` as the single-model benchmark.

Decision:

- Keep `lgbm_008_scale_110_submit` as the current best public submission.
- Do not overfit to public leaderboard feedback.
- Next work should be validation-only modeling or feature experiments, not more scale submissions.

---

## Experiment Feedback Summary

| exp_id | Result | Decision | Main Lesson |
|---|---:|---|---|
| `001_baseline_rf` | public 0.5879246832 | old baseline | RandomForest is useful only as a first sanity check. |
| `baseline_rf_001_valid_2024` | local 0.5777342168 | old baseline | Local validation pipeline works, but model is weak. |
| `lgbm_001` | local 0.5984976879 | useful baseline | LightGBM is much stronger than RandomForest. |
| `lgbm_001_submit_2025` | public 0.6024555766 | submitted baseline | Baseline LightGBM already generalizes reasonably. |
| `lgbm_002_wind` | local 0.5966447814 | no submission | Naive wind-derived vector features degraded performance. |
| `lgbm_003_tuned` | local 0.6033279875 | current best local | LightGBM tuning improved both NMAE and FICR. |
| `lgbm_003_tuned_submit` | public 0.60516 | previous best submitted | Main single-model comparison baseline. |
| `cat_001_baseline` | local 0.5980575665 | ensemble candidate only | CatBoost did not beat tuned LightGBM. |
| `xgb_001_baseline` | local 0.5988239498 | ensemble candidate only | XGBoost is the strongest non-LightGBM candidate so far, but still below tuned LightGBM. |
| `lgbm_004_weather_agg` | local 0.5962283588 | no submission | Broad row-wise weather aggregation added noise and hurt performance. |
| `lgbm_005_targeted_weather` | local 0.6019196756 | ensemble candidate only | Targeted weather features improved over broad aggregation but did not beat tuned LightGBM. |
| `ens_001_simple_avg` | local 0.6037465036 | submission candidate | A 50/50 average of `lgbm_003_tuned` and `lgbm_005_targeted_weather` beat the local benchmark. |
| `ens_001_simple_avg_submit` | public 0.6062263329 | previous best public submission | 50/50 ensemble improvement transferred to public leaderboard. |
| `ens_002_weight_search_cv` | local 0.6037465036 | validation-only reference | The best searched weight was again 0.50/0.50; no alternative weight beat the existing blend. |
| `ens_003_include_xgb_selective` | local 0.6037465036 | validation-only reference | Adding small XGB weights did not beat the current 50/50 LightGBM blend. |
| `lgbm_006_ficr_focus` | local 0.6109322217 | improved submission candidate | A simple 1.03 global scale improved both local total_score and FICR on the 2024 validation prediction. |
| `lgbm_006_ficr_focus_submit` | public 0.6158048399 | previous best public submission | The 1.03 scaled and capacity-clipped submission improved public total_score and FiCR strongly over `ens_001_simple_avg_submit`. |
| `lgbm_007_scale_robustness_cv` | local 0.6180968450 | possible submission candidate | Larger validation scales up to 1.08 kept improving FICR and total_score, so the upscaling effect is not isolated to 1.03. |
| `lgbm_008_scale_upper_sweep_cv` | local 0.6184267418 | converted to submission | The upper sweep peaked at scale 1.10 for both total_score and FICR, while larger scales degraded NMAE enough to lower total_score. |
| `lgbm_008_scale_110_submit` | public 0.6211026154 | current best public submission | The public gain over `lgbm_006_ficr_focus_submit` was FiCR-driven, while public 1-NMAE dropped. |
| `lgbm_009_multi_seed_lgbm_cv` | local 0.6179878399 | no submission | Multi-seed averaging improved stability but the best scaled score stayed below `lgbm_008` validation and lost 1-NMAE versus that reference. |
| `ens_004_group_scale_cv` | local 0.6248401656 | overfit risk confirmed | Group-wise scales 1.110/1.020/1.130 beat the global 1.10 validation reference on 2024, but the submission underperformed on public. |
| `ens_004_group_scale_submit` | public 0.6128355459 | not current best | Public score fell below `lgbm_008_scale_110_submit`, suggesting the 2024-only group-wise scale was overfit. |
| `cv_001_rolling_scale_robustness_cv` | 2024-only evaluated 0.6248401656 | insufficient rolling evidence | 2022 and 2023 were skipped by strict prior-year training rules, so no scale candidate proved robust across multiple years. |
| `lgbm_010_targeted_feature_v2_cv` | local 0.6184267418 scaled | no submission | The v2 feature set hurt standalone validation and did not improve any v2-including raw ensemble over `ens_001`; the scaled best only reproduced the existing global 1.10 reference. |
| `lgbm_011_objective_diversity_cv` | local 0.6214914951 scaled | future submission candidate | Objective diversity improved raw validation over `ens_001`, and conservative global scale 1.095 beat the `lgbm_008` validation reference without group-wise scaling. |

---

## Detailed Feedback by Experiment

### 1. RandomForest Baseline

Related experiments:

- `001_baseline_rf`
- `baseline_rf_001_valid_2024`

Feedback:

- RandomForest was useful as a first working baseline.
- It confirmed the data loading, feature creation, submission format, and metric pipeline.
- Performance was clearly lower than LightGBM, so it should not be a main modeling direction.

Decision:

- Keep for historical comparison.
- Do not spend more time tuning RandomForest.

---

### 2. `lgbm_001` - Baseline LightGBM

Feedback:

- LightGBM with baseline calendar + LDAPS/GFS mean features provided a major jump over RandomForest.
- This confirmed that gradient boosting is a strong fit for the current tabular weather-feature setup.
- The baseline feature matrix is valuable and should remain the default comparison point.

Decision:

- Keep as the first strong modeling baseline.
- Use later LightGBM variants to test whether new features or tuning truly improve over it.

---

### 3. `lgbm_002_wind` - Wind-Derived Feature Attempt

Feedback:

- Adding broad wind vector derivative features did not improve local validation.
- Local total_score dropped compared with `lgbm_001`.
- This suggests that naive wind-derived features can add noise or redundant information if not physically targeted.

Likely reasons:

- Derived wind features may have duplicated information already present in LDAPS/GFS variables.
- Feature construction may not have matched the actual wind-farm generation behavior.
- The model may have overfit noisy derived variables.

Decision:

- No submission.
- Do not add broad wind-derived features blindly.
- If wind features are revisited, use targeted physical features only, such as vertical wind shear or selected wind speed levels.

---

### 4. `lgbm_003_tuned` / `lgbm_003_tuned_submit` - Tuned LightGBM

Feedback:

- This is the previous best public submission and current single-model benchmark.
- Tuning LightGBM improved local validation and public leaderboard score.
- Smaller trees performed better:
  - `num_leaves=15`
  - `min_child_samples=20`
  - `reg_lambda=5.0`
- Larger tree settings such as `num_leaves=31` or `num_leaves=63` were generally less stable.

Decision:

- Previous best public submission and current single-model baseline.
- Keep as the main benchmark.
- Future experiments must beat `local total_score=0.6033279875` or provide clear ensemble diversity.

---

### 5. `cat_001_baseline` - CatBoost Baseline

Metrics:

| Metric | Value |
|---|---:|
| local total_score | 0.5980575665 |
| local one_minus_nmae | 0.8672565037 |
| local ficr | 0.3288586294 |

Feedback:

- CatBoost did not beat tuned LightGBM.
- Its NMAE component was close to LightGBM, but FICR was weaker.
- Since it is a different model family, it may still be useful in an ensemble.

Decision:

- No standalone submission.
- Keep as an ensemble diversity candidate.

---

### 6. `xgb_001_baseline` - XGBoost Baseline

Metrics:

| Metric | Value |
|---|---:|
| local total_score | 0.5988239498 |
| local one_minus_nmae | 0.8657252879 |
| local ficr | 0.3319226117 |

Feedback:

- XGBoost was slightly better than CatBoost on local total_score.
- It still did not beat tuned LightGBM.
- It is currently the strongest non-LightGBM model family tried.
- FICR is better than CatBoost but still below `lgbm_003_tuned`.

Decision:

- No standalone submission.
- Keep as an ensemble candidate.
- Do not spend too much time on XGBoost micro-tuning unless ensemble tests show value.

---

### 7. `lgbm_004_weather_agg` - Broad Row-Wise Weather Aggregation

Metrics:

| Metric | Value |
|---|---:|
| local total_score | 0.5962283588 |
| local one_minus_nmae | 0.8657741712 |
| local ficr | 0.3266825463 |
| beats `lgbm_003_tuned` | false |

Feedback:

- This experiment added row-wise LDAPS/GFS weather aggregations:
  - min
  - max
  - std
  - range
  - q25
  - q75
- It degraded performance compared with `lgbm_003_tuned`.
- The result should be recorded as a failed feature-expansion attempt.

Why the score likely dropped:

- The aggregation mixed weather variables with different physical meanings and units.
- Broad prefixes such as `ldaps_` and `gfs_` may combine wind, temperature, humidity, pressure, and other variables into the same summary statistics.
- Row-wise min/max/std/range over mixed-variable columns can create features that are numerically valid but physically unclear.
- The added features likely introduced noise rather than meaningful wind-power signals.
- Because the model already had mean-aggregated LDAPS/GFS features, these broad aggregations may have added redundant or misleading information.

Decision:

- No submission.
- Keep the experiment as evidence that broad, untargeted weather aggregation is not helpful.
- Do not continue with wide row-wise aggregation in this form.

---

### 8. `lgbm_005_targeted_weather` - Targeted Weather Features

Metrics:

| Metric | Value |
|---|---:|
| local total_score | 0.6019196756 |
| local one_minus_nmae | 0.8674731184 |
| local ficr | 0.3363662327 |
| beats `lgbm_003_tuned` | false |

Feedback:

- This experiment added explicit targeted weather features:
  - near-surface and hub-height wind speeds
  - simple GFS vertical-shear differences and ratios
  - GFS gust margins
  - LDAPS/GFS 10m wind component and speed differences
- It avoided broad row-wise LDAPS/GFS aggregations and avoided the wider `lgbm_002_wind` vector-derived feature set.
- The result recovered most of the loss from `lgbm_004_weather_agg` and improved `one_minus_nmae` slightly over `lgbm_003_tuned`.
- The lower FICR kept total score below the current best baseline.
- The 50/50 `ens_001_simple_avg` result proved that this weak standalone model still added useful ensemble diversity.

Decision:

- No standalone submission.
- Keep as an ensemble candidate because its NMAE behavior is competitive and the added features are physically interpretable.
- Do not expand targeted weather features broadly unless the next candidate has a clear physical reason and is checked against FICR.

---

### 9. `ens_001_simple_avg` - Validation-Only Simple Ensemble

Best candidate:

| Item | Value |
|---|---:|
| selected candidate | `ens_001_lgbm003_lgbm005_avg` |
| local total_score | 0.6037465036 |
| local one_minus_nmae | 0.8680552297 |
| local ficr | 0.3394377775 |
| beats `lgbm_003_tuned` | true |

Candidate scores:

| Candidate | total_score | one_minus_nmae | ficr | Decision |
|---|---:|---:|---:|---|
| `ens_001_lgbm003_lgbm005_avg` | 0.6037465036 | 0.8680552297 | 0.3394377775 | submission candidate |
| `ens_001_lgbm003_lgbm005_xgb_avg` | 0.6025990552 | 0.8678660421 | 0.3373320683 | possible submission candidate |
| `ens_001_lgbm003_lgbm005_cat_xgb_avg` | 0.6032029445 | 0.8682928764 | 0.3381130125 | possible submission candidate |
| `ens_001_weighted_lgbm003_lgbm005` | 0.6032676628 | 0.8679214256 | 0.3386138999 | possible submission candidate |
| `ens_001_weighted_lgbm003_lgbm005_xgb` | 0.6028452327 | 0.8679107443 | 0.3377797211 | possible submission candidate |

Feedback:

- The best validation-only ensemble is a simple 50/50 average of `lgbm_003_tuned` and `lgbm_005_targeted_weather`.
- It improved both `one_minus_nmae` and FICR slightly over the current local benchmark.
- Adding XGBoost and CatBoost increased model diversity but reduced FICR enough that total score stayed below the 2-model average.
- The result supports `lgbm_005_targeted_weather` as useful ensemble diversity even though it was weaker standalone.

Decision:

- Mark as a submission candidate by the predefined decision rule.
- Do not create final submission code until explicitly requested.
- If converting to a submission later, train/generate both member test predictions with separate reproducible scripts, average them 50/50, clip by capacity, and validate the final CSV.

---

### 10. `ens_001_simple_avg_submit` - Final Ensemble Submission File

Status:

- Created `submissions/ens_001_simple_avg.csv`.
- Created member targeted-weather test predictions at `outputs/predictions/lgbm_005_targeted_weather_test.csv`.
- Created ensemble test predictions at `outputs/predictions/ens_001_simple_avg_test.csv`.
- Saved run logs and targeted-weather model artifact.
- `python scripts/validate_submission.py submissions/ens_001_simple_avg.csv` passed.
- Uploaded to DACON as `ens_001_simple_avg edit` at 2026-07-09 03:27:38 KST.

Reference metrics:

| Metric | Value |
|---|---:|
| validation total_score | 0.6037465036 |
| validation one_minus_nmae | 0.8680552297 |
| validation ficr | 0.3394377775 |
| public total_score | 0.6062263329 |
| public one_minus_nmae | 0.8675678207 |
| public ficr | 0.3448848451 |
| public rank at submission time | 275 |

Comparison against previous best public reference:

| Metric | `lgbm_003_tuned_submit` | `ens_001_simple_avg_submit` |
|---|---:|---:|
| public total_score | 0.60516 | 0.6062263329 |
| public one_minus_nmae | 0.86678 | 0.8675678207 |
| public ficr | 0.34354 | 0.3448848451 |

Feedback:

- The local validation improvement transferred to the public leaderboard.
- Both public 1-NMAE and public FiCR improved over `lgbm_003_tuned_submit`.
- `lgbm_005_targeted_weather` was weaker as a standalone model, but useful in the 50/50 ensemble.
- Keep caution that final ranking depends on the private leaderboard.

Decision:

- Previous best public submission before `lgbm_006_ficr_focus_submit`.
- Use as the pre-scaling public ensemble reference for new ensemble candidates.

---

### 11. `ens_002_weight_search_cv` - Validation-Only Weight Search

Purpose:

Search weights between `lgbm_003_tuned` and `lgbm_005_targeted_weather` validation predictions without training new models, creating test predictions, or creating a submission CSV.

Best candidate:

| Item | Value |
|---|---:|
| selected candidate | `ens_002_lgbm003_0.50_lgbm005_0.50` |
| lgbm_003_tuned weight | 0.50 |
| lgbm_005_targeted_weather weight | 0.50 |
| local total_score | 0.6037465036 |
| local one_minus_nmae | 0.8680552297 |
| local ficr | 0.3394377775 |
| exact delta vs `ens_001` reference | +0.0000000000152984 |

Feedback:

- The search confirmed that the existing 50/50 blend is the best point among the tested coarse and fine grids.
- Nearby weights such as 0.60/0.40 and 0.65/0.35 remained above `lgbm_003_tuned` but did not beat the 50/50 blend.
- The exact score is microscopically above the stored `ens_001` reference due to numeric precision, but it is practically the same candidate and same weight.
- No new submission code should be created from this run alone.

Decision:

- Record as a validation-only reference.
- Keep `lgbm_006_ficr_focus_submit` as the current public reference and `ens_001_simple_avg_submit` as the pre-scaling ensemble reference.
- Use the result as evidence that further gains likely need another model family, selective ensembling, or FICR-aware calibration rather than simple two-model weight tuning.

---

### 12. `ens_003_include_xgb_selective` - Validation-Only Selective XGB Ensemble

Purpose:

Test whether adding `xgb_001_baseline` at small weights improves over the current 50/50 LightGBM ensemble.

Best candidate:

| Item | Value |
|---|---:|
| selected candidate | `ens_003_reference_lgbm003_lgbm005_50_50` |
| XGB weight | 0.00 |
| local total_score | 0.6037465036 |
| local one_minus_nmae | 0.8680552297 |
| local ficr | 0.3394377775 |
| XGB candidates beat reference | false |

Feedback:

- The 50/50 `lgbm_003_tuned` + `lgbm_005_targeted_weather` reference remained the best candidate.
- The best XGB candidate was `ens_003_xgb005_b` at total_score=0.6035233959, below the reference.
- Some XGB candidates still beat the single-model `lgbm_003_tuned` baseline, but none improved the current ensemble.
- Adding XGB reduced FICR versus the reference in every tested candidate, so this did not provide a FICR-diversity case.

Decision:

- Validation-only reference.
- Keep `lgbm_006_ficr_focus_submit` as the previous best public submission before `lgbm_008_scale_110_submit`.
- Do not create final submission code from this experiment.

---

### 13. `lgbm_006_ficr_focus` - Validation-Only FICR Postprocessing

Purpose:

Test whether simple postprocessing of `outputs/predictions/ens_001_simple_avg_valid_2024_best.csv` can improve local FICR or total_score without training models, creating test predictions, or creating a submission CSV.

Best candidate:

| Item | Value |
|---|---:|
| selected best by total_score | `lgbm_006_global_scale_103` |
| selected best by FICR | `lgbm_006_global_scale_103` |
| local total_score | 0.6109322217 |
| local one_minus_nmae | 0.8692902276 |
| local ficr | 0.3525742158 |
| beats `ens_001` validation total_score | true |
| beats `ens_001` validation FICR | true |

Feedback:

- A global 1.03 scale was the strongest tested postprocessing by both total_score and FICR.
- The improvement is large for a validation-only calibration, so it should be treated as a submission candidate but checked carefully for private-leaderboard robustness.
- The direction suggests the current validation ensemble may be underpredicting generation in a way that affects FICR thresholds.
- Conservative high-end shrink and downscaling hurt FICR, while small upscaling generally helped.

Decision:

- Mark as an improved submission candidate by the predefined validation rule.
- The postprocessing was converted into `lgbm_006_ficr_focus_submit`, which became the previous best public submission before `lgbm_008_scale_110_submit`.
- Keep private leaderboard robustness in mind before submitting more scaled variants.

---

### 14. `lgbm_006_ficr_focus_submit` - FICR Postprocessed Submission File

Status:

- Created `outputs/predictions/lgbm_006_ficr_focus_test.csv`.
- Created `submissions/lgbm_006_ficr_focus.csv`.
- Created `outputs/logs/lgbm_006_ficr_focus_submit.json`.
- Applied the validation-selected `lgbm_006_global_scale_103` postprocessing to `outputs/predictions/ens_001_simple_avg_test.csv`.
- Multiplied all target predictions by 1.03 and clipped each group to its capacity.
- Verified `forecast_id` and `forecast_kst_dtm` alignment with `data/raw/sample_submission.csv`.
- `python scripts/validate_submission.py submissions/lgbm_006_ficr_focus.csv` passed.

Reference metrics:

| Metric | Value |
|---|---:|
| validation total_score reference | 0.6109322217 |
| validation one_minus_nmae reference | 0.8692902276 |
| validation ficr reference | 0.3525742158 |
| public total_score | 0.6158048399 |
| public one_minus_nmae | 0.8679909923 |
| public ficr | 0.3636186875 |
| public rank at submission time | 245 |

Feedback:

- Uploaded to DACON as `lgbm_006_ficr_focus.csv edit` at 2026-07-09 10:20:19 KST.
- This is a postprocessing-only submission; no new models were trained.
- The source public reference was `ens_001_simple_avg_submit` with public total_score=0.6062263329, public one_minus_nmae=0.8675678207, public ficr=0.3448848451, and rank 275 at submission time.
- The 1.03 scaling improved public total_score by +0.0095785070, public FiCR by +0.0187338424, and public 1-NMAE by +0.0004231716 versus `ens_001_simple_avg_submit`.
- The public result confirms strong transfer for the validation-selected scaling, especially on FiCR.
- Caution: global 1.03 scaling may still be validation/public calibration, and the final target is the private leaderboard.

Decision:

- Current best public submission.
- Do not treat the public result alone as final proof of private-leaderboard robustness.

---

### 15. `lgbm_007_scale_robustness_cv` - Validation-Only Scale Robustness

Purpose:

Check whether the global scaling effect found in `lgbm_006_ficr_focus` is robust around scale 1.03 using `outputs/predictions/ens_001_simple_avg_valid_2024_best.csv`.

Best candidate:

| Item | Value |
|---|---:|
| selected best by total_score | 1.08 |
| selected best by FICR | 1.08 |
| local total_score | 0.6180968450 |
| local one_minus_nmae | 0.8684623965 |
| local ficr | 0.3677312936 |
| beats lgbm_006 scale 1.03 total_score | true |
| beats lgbm_006 scale 1.03 FICR | true |

Feedback:

- The validation upscaling effect is robust across the tested region, not just a one-point artifact at 1.03.
- Scales 1.035, 1.04, 1.05, 1.06, 1.07, and 1.08 all beat the lgbm_006 validation 1.03 reference on both total_score and FICR.
- In this sweep, FICR continued to improve through 1.08, while one_minus_nmae peaked around 1.04 and then declined slightly.
- The best validation scale 1.08 has a much higher FICR than 1.03, but it is also farther from the public-proven scale, so it carries public/private robustness risk.
- This run created only validation artifacts and did not create a submission CSV.

Decision:

- Mark as a possible submission candidate by validation rule.
- Keep `lgbm_006_ficr_focus_submit` as the current public best until a deliberate decision is made to spend another public submission attempt.
- If a larger scale is considered for submission later, prefer a conservative choice or a narrow follow-up validation-only sweep before creating final submission code.

---

### 16. `lgbm_008_scale_upper_sweep_cv` - Validation-Only Upper Scale Sweep

Purpose:

Extend the global scale robustness check above 1.08 using `outputs/predictions/ens_001_simple_avg_valid_2024_best.csv`.

Best candidate:

| Item | Value |
|---|---:|
| selected best by total_score | 1.10 |
| selected best by FICR | 1.10 |
| local total_score | 0.6184267418 |
| local one_minus_nmae | 0.8671491370 |
| local ficr | 0.3697043467 |
| beats `lgbm_007` scale 1.08 total_score | true |
| beats `lgbm_007` scale 1.08 FICR | true |

Feedback:

- Total_score and FICR both peaked at scale 1.10 in this sweep.
- Scale 1.095 was close behind and also beat the 1.08 reference on both total_score and FICR.
- Higher scales from 1.12 through 1.16 reduced one_minus_nmae enough that total_score declined.
- This is not an upper-bound artifact through 1.16, but scale 1.10 is still meaningfully more aggressive than the public-proven 1.03 submission.
- This run created only validation artifacts and did not create a submission CSV.

Decision:

- Mark scale 1.10 as a possible conservative submission candidate by validation rule.
- Do not create final submission code yet.
- Avoid another public submission unless the validation gain over 1.08 is judged meaningful enough against submission-budget and public-overfit risk.

---

### 17. `lgbm_008_scale_110_submit` - Scale 1.10 Submission File

Status:

- Created `outputs/predictions/lgbm_008_scale_110_test.csv`.
- Created `submissions/lgbm_008_scale_110.csv`.
- Created `outputs/logs/lgbm_008_scale_110_submit.json`.
- Applied scale 1.10 to `outputs/predictions/ens_001_simple_avg_test.csv`.
- Clipped each target to group capacity.
- Verified `forecast_id` and `forecast_kst_dtm` alignment with `data/raw/sample_submission.csv`.
- `python scripts/validate_submission.py submissions/lgbm_008_scale_110.csv` passed.
- Uploaded to DACON as `0710_v1 edit` at 2026-07-10 17:54:24 KST.

Reference metrics:

| Metric | Value |
|---|---:|
| validation total_score reference | 0.6184267418 |
| validation one_minus_nmae reference | 0.8671491370 |
| validation ficr reference | 0.3697043467 |
| public total_score | 0.6211026154 |
| public one_minus_nmae | 0.8634833072 |
| public ficr | 0.3787219236 |
| public rank at submission time | 148 |

Feedback:

- This is a postprocessing-only submission; no new models were trained.
- The selected 1.10 scale came from `lgbm_008_scale_upper_sweep_cv`, where it beat the `lgbm_007` 1.08 scale on both total_score and FICR.
- Scale 1.10 was the peak in the 1.075-1.16 upper sweep, not the upper boundary.
- This submission is more aggressive than the 1.03 scale used by `lgbm_006_ficr_focus_submit`.
- Public total_score improved by +0.0052977755 versus `lgbm_006_ficr_focus_submit`.
- Public FiCR improved by +0.0151032361 versus `lgbm_006_ficr_focus_submit`.
- Public 1-NMAE decreased by -0.0045076851 versus `lgbm_006_ficr_focus_submit`.
- The public score increase is mainly FiCR-driven.
- Keep caution that final ranking depends on the private leaderboard.

Decision:

- Current best public submission.
- Keep `lgbm_008_scale_110_submit` as the public reference.
- Avoid more public probing today; next work should be validation-only modeling or feature experiments, not more scale submissions.

---

### 18. `lgbm_009_multi_seed_lgbm_cv` - Validation-Only Multi-Seed LightGBM

Purpose:

Test whether averaging tuned `lgbm_003` LightGBM models across seeds improves validation stability and 1-NMAE before any future submission generation.

Best candidate:

| Item | Value |
|---|---:|
| selected best scale | 1.10 |
| local total_score | 0.6179878399 |
| local one_minus_nmae | 0.8668335806 |
| local ficr | 0.3691420992 |
| raw ensemble total_score | 0.6023894779 |
| beats `lgbm_008` validation total_score | false |

Feedback:

- The raw multi-seed average did not improve over the existing ensemble references.
- Scaling recovered FICR, but the best scaled candidate remained below `lgbm_008_scale_upper_sweep_cv` total_score=0.6184267418.
- The best scaled candidate also reduced 1-NMAE versus the `lgbm_008` validation reference.
- This is not a strong candidate for submission generation.

Decision:

- Keep `lgbm_008_scale_110_submit` as the current public reference.
- Do not create submission code or test predictions from this experiment.

---

### 19. `ens_004_group_scale_cv` - Validation-Only Group-Wise Scaling

Purpose:

Search separate scale factors for `kpx_group_1`, `kpx_group_2`, and `kpx_group_3` using `outputs/predictions/ens_001_simple_avg_valid_2024_best.csv`.

Best candidate:

| Item | Value |
|---|---:|
| selected scales | 1.110 / 1.020 / 1.130 |
| local total_score | 0.6248401656 |
| local one_minus_nmae | 0.8696017636 |
| local ficr | 0.3800785675 |
| delta total vs `lgbm_008` validation | +0.0064134238 |
| delta one_minus_nmae vs `lgbm_008` validation | +0.0024526266 |
| delta ficr vs `lgbm_008` validation | +0.0103742208 |

Feedback:

- Coarse grid best was `1.105 / 1.030 / 1.120` with total_score=0.6240789838.
- Fine grid best was `1.110 / 1.020 / 1.130`, improving both total_score and 1-NMAE versus the global 1.10 validation reference.
- This is stronger than a pure FiCR tradeoff because 1-NMAE is preserved and improved.

Decision:

- Mark as a strong future submission candidate by the predefined rule.
- No test predictions or submission CSV were created in this validation-only run.

---

### 20. `ens_004_group_scale_submit` - Group-Wise Scale Submission File

Status:

- Created `outputs/predictions/ens_004_group_scale_test.csv`.
- Created `submissions/ens_004_group_scale.csv`.
- Created `outputs/logs/ens_004_group_scale_submit.json`.
- Applied group-wise scales 1.110 / 1.020 / 1.130 to `outputs/predictions/ens_001_simple_avg_test.csv`.
- Clipped each target to group capacity.
- Verified `forecast_id` and `forecast_kst_dtm` alignment with `data/raw/sample_submission.csv`.
- `python scripts/validate_submission.py submissions/ens_004_group_scale.csv` passed.

Reference metrics:

| Metric | Value |
|---|---:|
| validation total_score reference | 0.6248401656 |
| validation one_minus_nmae reference | 0.8696017636 |
| validation ficr reference | 0.3800785675 |
| public total_score | 0.6128355459 |
| public one_minus_nmae | 0.8625281665 |
| public ficr | 0.3631429254 |

Feedback:

- This is a postprocessing-only submission; no new models were trained.
- Validation improved total_score by +0.0064134238 versus the global 1.10 validation reference.
- Validation improved 1-NMAE by +0.0024526266 versus the global 1.10 validation reference.
- Validation improved FiCR by +0.0103742208 versus the global 1.10 validation reference.
- Uploaded to DACON as `ens_004_group_scale edit` at 2026-07-11 16:04:53 KST.
- The public score underperformed `lgbm_008_scale_110_submit` public total_score=0.6211026154.
- This result suggests the group-wise scale was likely overfit to the single 2024 validation split.
- Keep caution that final ranking depends on the private leaderboard, and public feedback should not be overused for model selection.

Decision:

- Not current best public submission.
- Do not submit variants of this exact 2024-selected group-wise scale without stronger time-aware validation evidence.
- Keep `lgbm_008_scale_110_submit` as the current public reference.

---

### 21. `cv_001_rolling_scale_robustness_cv` - Rolling-Year Scale Robustness

Purpose:

Evaluate global and group-wise scaling candidates across requested validation years 2022, 2023, and 2024 instead of relying only on the 2024 split.

Fold status:

| validation_year | status | reason |
|---:|---|---|
| 2022 | skipped | no training rows with forecast year < 2022 |
| 2023 | skipped | `kpx_group_3` had 0 prior training labels |
| 2024 | evaluated | valid strict rolling fold |

Key results:

| Candidate | mean_total_score | min_total_score | mean_1_nmae | mean_ficr | evaluated_years |
|---|---:|---:|---:|---:|---|
| `ens_004_failed_group_scale_1p110_1p020_1p130` | 0.6248401656 | 0.6248401656 | 0.8696017636 | 0.3800785675 | 2024 |
| `global_1p1` | 0.6184267418 | 0.6184267418 | 0.8671491370 | 0.3697043467 | 2024 |
| `raw_no_scale` | 0.6037465036 | 0.6037465036 | 0.8680552297 | 0.3394377775 | 2024 |

Feedback:

- The script successfully recreated the strongest validation source as a 50/50 raw ensemble of the tuned baseline LightGBM and targeted-weather LightGBM.
- Under strict rolling-year rules, only 2024 was evaluable because earlier folds lacked prior target history for all groups.
- The failed `ens_004` group scale remained best on the 2024 fold, but this is exactly the split where it was selected.
- The experiment could not prove instability across years because only one fold was evaluable.
- The public underperformance of `ens_004_group_scale_submit` remains the practical evidence that the group-wise scale overfit the 2024 validation split.

Decision:

- Insufficient rolling-year evidence for a robust scale candidate.
- Keep `lgbm_008_scale_110_submit` as the current best public reference.
- Shift next work to feature/model improvements rather than more scale variants.

---

### 22. `lgbm_010_targeted_feature_v2_cv` - Targeted Feature v2 Validation

Purpose:

Test targeted row-local meteorological and time features on top of the existing `lgbm_005_targeted_weather` feature surface, without creating test predictions or a submission CSV.

Key results:

| Candidate | total_score | one_minus_nmae | ficr |
|---|---:|---:|---:|
| `lgbm_003_tuned_baseline_recheck` | 0.6033279875 | 0.8673265858 | 0.3393293893 |
| `lgbm_005_targeted_weather_recheck` | 0.6019196756 | 0.8674731184 | 0.3363662327 |
| `lgbm_010_targeted_feature_v2` | 0.5980917069 | 0.8663548879 | 0.3298285259 |
| best v2-including raw ensemble | 0.6035379179 | 0.8679159306 | 0.3391599052 |
| best raw overall, old 0.5/0.5 blend | 0.6037465036 | 0.8680552297 | 0.3394377775 |
| best conservative scaled candidate | 0.6184267418 | 0.8671491370 | 0.3697043467 |

Feedback:

- The script created 37 v2 features and skipped wind-direction sin/cos because no direction columns existed.
- The standalone v2 model reduced both 1-NMAE and FiCR versus the baseline and targeted-weather rechecks.
- The best v2-including ensemble used only 0.10 weight on v2 and still stayed below `ens_001_simple_avg`.
- The selected scaled candidate was the old `ens_001` 0.5/0.5 raw blend at global scale 1.10, matching the existing `lgbm_008` validation reference rather than improving it.

Decision:

- Do not create submission code from this experiment.
- Keep `lgbm_008_scale_110_submit` as the current best public reference.
- Continue feature/model experiments, but avoid adding more broad derived weather features without a clearer physical reason.

---

### 23. `lgbm_011_objective_diversity_cv` - LightGBM Objective Diversity Validation

Purpose:

Train LightGBM variants with different objective behavior on the existing safe baseline and targeted-weather feature sets, then test raw objective-diversity ensembles before any future submission generation.

Key results:

| Candidate | total_score | one_minus_nmae | ficr |
|---|---:|---:|---:|
| best single objective, `baseline_regression_l1` | 0.6052244336 | 0.8670253828 | 0.3434234844 |
| reference 0.5 baseline regression + 0.5 targeted regression | 0.6037465036 | 0.8680552297 | 0.3394377775 |
| best raw objective-diversity ensemble | 0.6077784238 | 0.8683304324 | 0.3472264153 |
| best conservative scaled candidate, scale 1.095 | 0.6214914951 | 0.8674785360 | 0.3755044543 |

Feedback:

- `regression_l1` was the strongest single-objective variant on both baseline and targeted feature sets.
- Huber performed poorly with the current base parameters and should not be used as-is.
- Poisson and Tweedie trained successfully because labels were nonnegative, but both were weak standalone candidates.
- The best raw candidate used top-4 weight search with weights `baseline_regression_l1=0.3`, `targeted_regression_l1=0.6`, and `targeted_regression=0.1`.
- The raw candidate beat `ens_001_simple_avg` validation total_score=0.6037465036, so objective diversity is useful by the predefined rule.
- The best conservative scaled candidate beat `lgbm_008` validation total_score=0.6184267418 without group-wise scale.

Decision:

- Mark as a future submission candidate by validation rule.
- No test predictions or submission CSV were created.
- Keep `lgbm_008_scale_110_submit` as current best public reference until a deliberate submission-budget decision is made.

---

## Main Lessons So Far

### 1. Validation setup matters

- Continue using 2024 time-based local validation.
- Avoid random validation splits because this is a time-series forecasting-style competition.
- Do not use future information after the prediction cutoff time.

### 2. LightGBM is currently the strongest model family

- RandomForest is too weak.
- CatBoost and XGBoost are useful for ensemble diversity but do not beat tuned LightGBM individually.
- Current best public submission is `lgbm_008_scale_110_submit`.
- `lgbm_003_tuned_submit` remains the strongest single-model baseline and comparison point.

### 3. Feature engineering must be physically meaningful

Failed or weak directions:

- Broad wind-derived features without careful targeting.
- Broad row-wise LDAPS/GFS aggregation across mixed weather variables.

More promising direction:

- Targeted weather features based on wind-power domain logic.
- Targeted weather features may help NMAE but still need FICR-aware validation.

### 4. FICR is important

- Some models have decent NMAE but weak FICR.
- Experiments should be judged by total_score, but FICR changes should be tracked carefully.
- A model with slightly lower total_score but meaningfully different FICR behavior may still have ensemble value.
- In `ens_001_simple_avg`, the 2-model average improved FICR slightly, while adding CatBoost/XGBoost reduced FICR.

### 5. Do not overfit to public leaderboard

- Public score is useful feedback, but not the final private leaderboard.
- Use `lgbm_008_scale_110_submit` as the public reference, `lgbm_006_ficr_focus_submit` as the previous scaled public reference, `ens_001_simple_avg_submit` as the pre-scaling ensemble reference, and `lgbm_003_tuned_submit` as the single-model reference, but keep validation logic time-aware.
- Treat `ens_004_group_scale_submit` as a public underperformance case and avoid variants of the same 2024-selected group-wise scale.
- Record failed experiments because they are useful for later reports and final presentation.

---

## Completed Targeted Weather Experiment

### `lgbm_005_targeted_weather`

Purpose:

Test physically meaningful weather features instead of broad row-wise aggregations.

Recommended feature direction:

1. Wind-speed-related features
   - 10m wind speed
   - 80m wind speed
   - 100m wind speed
   - selected GFS/LDAPS wind variables if available

2. Vertical wind shear
   - `wind_speed_100m - wind_speed_10m`
   - `wind_speed_80m - wind_speed_10m`
   - ratios only if numerically stable

3. Time cyclic features
   - `hour_sin`
   - `hour_cos`
   - `month_sin`
   - `month_cos`

4. Carefully selected weather differences
   - GFS high-altitude wind minus near-surface wind
   - LDAPS/GFS differences for comparable variables only, if names clearly match

Avoid:

- Mixing unrelated weather variables into one aggregate.
- Broad prefix-level min/max/std over all `ldaps_` or all `gfs_` columns.
- Group target historical averages unless leakage is carefully controlled.
- Any feature that uses future information after the prediction cutoff.

Success criteria:

| Result | Decision |
|---|---|
| local total_score > 0.6033279875 | Create submission candidate |
| local total_score 0.601-0.603 | Keep as ensemble candidate |
| local total_score < 0.600 | Record only |
| FICR improves clearly | Consider ensemble even if total_score is slightly lower |

Outcome:

- Local total_score was 0.6019196756, below `lgbm_003_tuned`.
- `one_minus_nmae` improved slightly, but FICR dropped.
- Keep as an ensemble candidate only.
- It was weak standalone but useful in the 50/50 `ens_001_simple_avg` ensemble, which became the previous best public submission before `lgbm_006_ficr_focus_submit`.

---

## Current Action Plan

1. Keep `lgbm_008_scale_110_submit` as the current public reference.
2. Record `lgbm_004_weather_agg` as a failed broad aggregation attempt.
3. Keep `lgbm_005_targeted_weather` as an ensemble candidate, not a standalone submission.
4. Keep `ens_002_weight_search_cv` as a validation-only reference; the 50/50 blend remains best among searched weights.
5. Keep `ens_003_include_xgb_selective` as a validation-only reference; small XGB weights did not improve the blend.
6. `lgbm_008_scale_110_submit` is the current best public submission after public total_score=0.6211026154.
7. The public gain over `lgbm_006_ficr_focus_submit` was FiCR-driven, while public 1-NMAE dropped.
8. `ens_004_group_scale_submit` underperformed on public total_score=0.6128355459, likely due to 2024 validation overfit.
9. `cv_001_rolling_scale_robustness_cv` did not prove a robust multi-year scale candidate because only 2024 was evaluable.
10. Avoid more public probing today.
11. `lgbm_010_targeted_feature_v2_cv` did not improve raw validation, so continue validation-only feature/model experiments.

---

## Submission Policy

Submit only when:

- Local validation beats `lgbm_003_tuned`, or
- The model has strong ensemble potential with meaningfully different prediction behavior, or
- A final ensemble candidate is ready.

Do not submit:

- `cat_001_baseline` alone.
- `xgb_001_baseline` alone.
- `lgbm_004_weather_agg`.
