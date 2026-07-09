# Experiment Feedback

Updated: 2026-07-09 KST

This file records qualitative feedback, experiment interpretation, failure analysis, and next actions for the DACON wind power generation forecasting project.

Use this file together with `docs/experiment_log.md`:

- `docs/experiment_log.md`: numeric experiment/submission history
- `docs/experiment_feedback.md`: interpretation, lessons, and next experiment decisions

---

## Current Reference Baseline

### `lgbm_006_ficr_focus_submit`

Current best public submission.

| Item | Value |
|---|---:|
| local total_score | 0.6109322217 |
| local one_minus_nmae | 0.8692902276 |
| local ficr | 0.3525742158 |
| DACON public total_score | 0.6158048399 |
| DACON public one_minus_nmae | 0.8679909923 |
| DACON public ficr | 0.3636186875 |
| public rank at submission time | 245 |

Key interpretation:

- The 1.03 globally scaled and capacity-clipped `ens_001_simple_avg_test` prediction is the current public reference.
- The validation-selected 1.03 global scaling transferred strongly to the public leaderboard.
- Public total_score improved by +0.0095785070 versus `ens_001_simple_avg_submit`.
- Public FiCR improved by +0.0187338424 versus `ens_001_simple_avg_submit`.
- Public 1-NMAE also improved by +0.0004231716 versus `ens_001_simple_avg_submit`.
- `lgbm_005_targeted_weather` was weak standalone but useful in the 50/50 ensemble.
- Keep caution that the final target is the private leaderboard, not public score alone.
- Avoid excessive public probing because repeated public feedback can overfit model selection to the public split.
- `lgbm_003_tuned_submit` remains the main single-model comparison baseline.
- Previous best public reference: `ens_001_simple_avg_submit` with public total_score=0.6062263329, public one_minus_nmae=0.8675678207, public ficr=0.3448848451, rank 275 at submission time.
- Previous best public reference: `lgbm_003_tuned_submit` with public total_score=0.60516, public one_minus_nmae=0.86678, public ficr=0.34354, rank 277 at submission time.
- The best LightGBM setting used a relatively small tree structure:
  - `num_leaves=15`
  - `min_child_samples=20`
  - `learning_rate=0.03`
  - `n_estimators=1000`
  - `reg_lambda=5.0`
- Smaller, more regularized trees appear better than larger, more complex trees for the current feature set.
- Use `lgbm_006_ficr_focus_submit` as the public reference, `ens_001_simple_avg_submit` as the pre-scaling ensemble reference, and `lgbm_003_tuned_submit` as the single-model benchmark.

Decision:

- Keep `lgbm_006_ficr_focus_submit` as the current best public submission.
- Do not overfit to public leaderboard feedback.

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
| `lgbm_006_ficr_focus_submit` | public 0.6158048399 | current best public submission | The 1.03 scaled and capacity-clipped submission improved public total_score and FiCR strongly over `ens_001_simple_avg_submit`. |

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
- Uploaded to DACON as `ens_001_simple_avg edit` by 배추도사님 at 2026-07-09 03:27:38 KST.

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
- Keep `lgbm_006_ficr_focus_submit` as the current best public submission.
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
- The postprocessing was converted into `lgbm_006_ficr_focus_submit` and is now the current best public submission.
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

- Uploaded to DACON as `lgbm_006_ficr_focus.csv edit` by 배추도사님 at 2026-07-09 10:20:19 KST.
- This is a postprocessing-only submission; no new models were trained.
- The source public reference was `ens_001_simple_avg_submit` with public total_score=0.6062263329, public one_minus_nmae=0.8675678207, public ficr=0.3448848451, and rank 275 at submission time.
- The 1.03 scaling improved public total_score by +0.0095785070, public FiCR by +0.0187338424, and public 1-NMAE by +0.0004231716 versus `ens_001_simple_avg_submit`.
- The public result confirms strong transfer for the validation-selected scaling, especially on FiCR.
- Caution: global 1.03 scaling may still be validation/public calibration, and the final target is the private leaderboard.

Decision:

- Current best public submission.
- Do not treat the public result alone as final proof of private-leaderboard robustness.

---

## Main Lessons So Far

### 1. Validation setup matters

- Continue using 2024 time-based local validation.
- Avoid random validation splits because this is a time-series forecasting-style competition.
- Do not use future information after the prediction cutoff time.

### 2. LightGBM is currently the strongest model family

- RandomForest is too weak.
- CatBoost and XGBoost are useful for ensemble diversity but do not beat tuned LightGBM individually.
- Current best public submission is `lgbm_006_ficr_focus_submit`.
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
- Use `lgbm_006_ficr_focus_submit` as the public reference, `ens_001_simple_avg_submit` as the pre-scaling ensemble reference, and `lgbm_003_tuned_submit` as the single-model reference, but keep validation logic time-aware.
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

1. Keep `lgbm_006_ficr_focus_submit` as the current best public reference.
2. Record `lgbm_004_weather_agg` as a failed broad aggregation attempt.
3. Keep `lgbm_005_targeted_weather` as an ensemble candidate, not a standalone submission.
4. Keep `ens_002_weight_search_cv` as a validation-only reference; the 50/50 blend remains best among searched weights.
5. Keep `ens_003_include_xgb_selective` as a validation-only reference; small XGB weights did not improve the blend.
6. `lgbm_006_ficr_focus_submit` is the current best public submission after public total_score=0.6158048399.
7. Next consider validation-only robustness checks for the global upscaling effect, such as 1.01/1.02/1.03/1.04, or new FICR-focused modeling before any further public submission.

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
