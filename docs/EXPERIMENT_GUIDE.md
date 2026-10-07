# Experiment Guide

This guide summarizes modeling questions and the scripts that address them. It presents qualitative decisions; the original dated experiment records remain in [experiment_log.md](experiment_log.md) and [experiment_feedback.md](experiment_feedback.md).

## Experiment Map

| Question | Scripts | Interpretation |
|---|---|---|
| Establish a temporal baseline | `train_baseline_rf_cv.py`, `train_lgbm_cv.py` | Establish a common feature frame and split before adding complexity |
| Tune a tree model | `tune_lgbm_cv.py` | Compare parameter candidates on the same local holdout |
| Add broad wind features | `train_lgbm_wind_cv.py` | Test speed/direction derivatives rather than assuming physical relevance guarantees improvement |
| Summarize weather columns | `train_lgbm_weather_agg_cv.py` | Examine whether broad row summaries lose useful distinctions between variables |
| Add targeted weather features | `train_lgbm_targeted_weather_cv.py`, `train_lgbm_targeted_feature_v2_cv.py` | Compare height-specific wind and cross-source signals |
| Compare model families | `train_catboost_cv.py`, `train_xgb_cv.py` | Evaluate alternative learners on the same baseline feature surface |
| Combine predictions | `evaluate_simple_ensemble_cv.py`, `evaluate_weight_search_ensemble_cv.py`, `evaluate_selective_xgb_ensemble_cv.py` | Check whether diversity improves a combination of aligned predictions |
| Calibrate global outputs | `evaluate_ficr_postprocess_cv.py`, `evaluate_scale_robustness_cv.py`, `evaluate_scale_upper_sweep_cv.py` | Study the forecast-error/settlement tradeoff without retraining |
| Add calibration flexibility | `evaluate_group_scale_cv.py` | Evaluate separate group multipliers and their overfitting risk |
| Examine multiple periods | `evaluate_rolling_scale_robustness_cv.py` | Report both fold results and reasons a fold cannot be evaluated |
| Add training diversity | `train_lgbm_multiseed_cv.py`, `train_lgbm_objective_diversity_cv.py` | Compare seed averaging and objective variants before generating new submissions |

All script paths are relative to `scripts/`. Evaluation scripts often consume predictions from earlier runs; inspect their configured input paths before execution.

## Decisions Retained from Development

### Targeted Features and Ensembles

The targeted-weather variant focuses on named wind component pairs, height differences, gust margins and LDAPS/GFS differences. It provides an alternative feature view for combination with the tuned baseline.

A model's standalone behavior and its usefulness in an ensemble are different questions. Compare the combined predictions using the same targets, row order and time split.

### Broad Aggregates

Means across grid rows and statistics across different weather columns are separate operations. The latter may combine variables with different units. Changes should document which variable groups are summarized and why those summaries have a useful interpretation.

### Scaling and Robustness

Scaling affects both absolute normalized error and the settlement component. A candidate can improve one while worsening the other. Clipping also changes how a multiplier behaves near group capacity.

Group-wise calibration offers more flexibility than one global scale. The recorded group-scale submission did not transfer as expected from the local holdout, so its local selection should be treated as an overfitting case rather than a general rule.

### Rolling-Year Coverage

The rolling script requests validation years 2022, 2023 and 2024. Original development notes report that the first two folds were skipped because usable prior training history was missing for the full set of targets. Only 2024 was evaluated.

That run therefore provides one evaluated fold, not evidence of robustness across three years. A future rolling design should report training coverage and missing-label coverage per target, along with every skipped-fold reason.

## Extending an Experiment

1. State one hypothesis and identify the existing comparison run.
2. Keep the forecast cutoff and validation period explicit.
3. Choose a new experiment/run identifier and inspect all output paths.
4. Change one model, feature or calibration choice at a time.
5. Fit imputation and model parameters using training rows only.
6. Compare error and settlement behavior by target, including clipping frequency.
7. Record input windows, feature columns, seeds, parameters and generated relative paths.
8. Generate a submission only after the validation comparison and alignment checks are understood.

Use the existing artifact layout in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Preserve earlier experiment records and predictions when creating a new run.

## Useful Next Development Steps

| Extension | Observable outcome |
|---|---|
| Cutoff-aware weather selection | Every retained forecast record is eligible at the prediction cutoff |
| Configurable run identifiers | Repeated experiments write to distinct paths without editing many constants |
| Explicit fold coverage reports | Every evaluated/skipped fold has per-target training counts and a reason |
| Reusable prediction alignment | Model and postprocessing routes share one row-identity check |
| Separate saved-model inference | Inference can use a stored imputer, feature order and model bundle |

These are extension proposals. The current scripts implement the recorded training, prediction and evaluation routes described in this documentation.
