# Model Design

This guide follows the implemented code. The main submission route uses tuned and targeted-weather LightGBM models; additional scripts explore other model families and postprocessing choices.

## Data Flow and Responsibilities

| Stage | Module or scripts | Input and output contract |
|---|---|---|
| Local data loading | [data.py](../src/dacon_wind/data.py) | Read labels/weather/sample CSVs and parse timestamps |
| Weather aggregation | [features.py](../src/dacon_wind/features.py) | Reduce grid rows to one mean feature row per forecast timestamp |
| Feature join | `build_baseline_train_frame`, `build_baseline_test_frame` | Join forecast features to label or sample rows |
| Time split | [cv.py](../src/dacon_wind/cv.py) | Boolean masks for years before 2024 and year 2024 |
| Training | [models.py](../src/dacon_wind/models.py), `train_*_cv.py`, `train_*_submit.py` | Training-fitted imputer and separate regressors for each target |
| Evaluation | [metric.py](../src/dacon_wind/metric.py) | Group error, settlement component and combined objective |
| Ensemble/calibration | `evaluate_*_cv.py`, `create_*_submission.py` | Aligned prediction tables and capacity-clipped outputs |
| Submission validation | [validate_submission.py](../scripts/validate_submission.py) | Errors list or CLI success/failure exit code |

## Timestamp and Join Behavior

Generation labels use `kst_dtm`; feature construction renames it to `forecast_kst_dtm`. The sample already supplies forecast timestamps.

LDAPS and GFS are separately grouped by forecast timestamp and averaged across grid rows. Coordinates, grid identifiers and `data_available_kst_dtm` are excluded from the mean features. The two aggregated weather tables are inner-joined; that combined table is left-joined to labels or the sample.

Consequences of these joins:

- A timestamp must appear in both weather sources to enter the combined weather table.
- Label/sample rows remain after the left join, even when their weather features are missing.
- Later median imputation handles missing values in the feature matrix.
- Grid-level geography is not retained by this mean-aggregation baseline.

The current aggregation does not choose forecast records by their availability time. An availability-aware loader should select records before the applicable prediction cutoff, then aggregate them. Timestamp alignment alone does not enforce that cutoff.

## Feature Families

### Calendar and Baseline Weather

`calendar_features` produces month, day, hour, day of week and weekend indicators. It also encodes hour and month as sine/cosine pairs:

```text
hour_sin = sin(2 * pi * hour / 24)
hour_cos = cos(2 * pi * hour / 24)
month_sin = sin(2 * pi * month / 12)
month_cos = cos(2 * pi * month / 12)
```

`build_feature_matrix` excludes the timestamp, forecast identifier and all target columns, then combines calendar features with the remaining weather columns. This keeps target labels out of model input.

### Broad Wind Derivatives

`build_wind_feature_matrix` adds speed, normalized direction components, squared speed and cubed speed for named LDAPS/GFS component pairs. It includes near-surface, higher-altitude and pressure-level pairs.

For horizontal components `u` and `v`, speed is `sqrt(u*u + v*v)`. Direction components divide by speed; zero-speed rows use zero after missing-value replacement. This broader variant raises an error when required component columns are absent.

### Row-Wise Weather Aggregates

`build_weather_agg_feature_matrix` adds min, max, standard deviation, range and quartiles across selected already-aggregated weather columns within each timestamp row.

This is different from the earlier mean across spatial grid rows. Broad column groups can contain variables with different units, so the physical interpretation depends on the selected group.

### Targeted Weather Features

`build_targeted_weather_feature_matrix` adds selected wind-speed features, vertical differences and ratios, gust margins, and LDAPS/GFS near-surface differences.

Source pairs include LDAPS 10 m and GFS 10/80/100 m wind components. A targeted feature is created only when its required columns exist. Near-zero denominators in speed ratios become missing values and are handled by the training-fitted imputer.

A speed difference between heights is a shear-related feature; it is not a fitted physical turbine power curve or a complete atmospheric shear model.

## Training and Prediction

The main LightGBM route uses the following sequence:

1. Build the training and validation/test feature matrices.
2. Fit a median `SimpleImputer` on training features.
3. Transform validation/test features using that fitted imputer.
4. For each target, select rows with non-null training labels.
5. Fit an independent regressor and predict the requested rows.
6. Clip predictions to the target's allowed range.

| Target | Capacity upper bound, kWh |
|---|---:|
| `kpx_group_1` | 21600 |
| `kpx_group_2` | 21600 |
| `kpx_group_3` | 21000 |

The shared training result keeps the imputer, model dictionary, ordered feature columns, predictions and per-target training row counts. Final training scripts save corresponding model bundles and run summaries.

Validation fits models on years before 2024. Submission scripts refit on all available training labels and reindex test features to the training column order. The submission route contains its selected parameters directly; it does not require rerunning the tuning script.

## Evaluation Formula

The metric first selects, separately for each group, rows whose actual generation is at least 10% of capacity. Other rows are excluded from that group's evaluation; this eligibility rule does not remove low-generation rows from model training.

For each eligible row:

```text
normalized_error = abs(forecast - actual) / group_capacity

settlement_weight =
    4 if normalized_error <= 0.06
    3 if normalized_error <= 0.08
    0 otherwise

group_nmae = mean(normalized_error)
group_ficr = sum(actual * settlement_weight) / sum(actual * 4)

one_minus_nmae = 1 - mean(group_nmae across groups)
ficr = mean(group_ficr across groups)
combined_objective = 0.5 * one_minus_nmae + 0.5 * ficr
```

The group average gives each target equal weight, while the settlement calculation inside each group is weighted by actual generation. The metric consumes target rows in their supplied order; callers must align actuals and predictions before passing them in.

The implementation rejects missing target columns, unequal row counts and groups with no eligible actual-generation rows. It does not align rows by forecast ID automatically.

## Ensemble and Calibration Contracts

The recorded ensemble averages the tuned and targeted-weather predictions with equal weights. Each member must match the sample's row count, `forecast_id` and parsed `forecast_kst_dtm` order before combination.

Global scaling applies the same multiplier to each group's ensemble prediction, followed by group-capacity clipping. The scale-1.10 script reads the scale-1.03 submission as a comparison/alignment reference, but scales the original ensemble directly.

The CSV validator checks table shape, exact column order, missing values, numeric target values, non-negativity and capacity bounds. The validator itself does not compare forecast ID/timestamp values to the sample; ensemble and postprocessing functions perform those additional checks.

## Design Extensions

Useful next changes are an explicit availability-cutoff selection layer, configurable run IDs for repeatable experiment isolation, and validation folds that report coverage for every target. Implement each as a separate change with a clear before/after data contract and a comparable time split.
