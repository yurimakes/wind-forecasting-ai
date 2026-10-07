# Wind Power Forecasting

## Overview

Experiment code for the [DACON BARAM 2026 wind power forecasting competition](https://dacon.io/competitions/official/236727/overview/description). LDAPS/GFS weather forecasts and calendar features are used to predict hourly generation for three KPX groups.

The recorded approach compares RandomForest, LightGBM, CatBoost and XGBoost, then combines two LightGBM variants and applies capacity-clipped scaling. This repository documents the experiments, including unsuccessful feature changes.

This repository focuses on the modeling workflow, experiment design and lessons learned. Raw-data training and inference have not been independently rerun for this documentation update.

Main workflow:

1. Load raw train/test weather and label data.
2. Build baseline calendar + LDAPS/GFS mean features.
3. Use 2024 time-based local validation.
4. Train separate models for `kpx_group_1`, `kpx_group_2`, and `kpx_group_3`.
5. Clip predictions by each group capacity.
6. Validate every submission using `scripts/validate_submission.py`.

Raw data, trained models, row-level predictions and submission CSVs are excluded. Keep authorized local data under `data/raw/`, models under `outputs/models/`, predictions under `outputs/predictions/`, logs under `outputs/logs/`, and submission CSVs under `submissions/`. See [sources and usage conditions](docs/SOURCES.md).

## Modeling Approach and Lessons

| Component | Approach |
|---|---|
| Baselines | RandomForest, LightGBM, CatBoost and XGBoost |
| Features | Calendar features and mean-aggregated LDAPS/GFS forecasts |
| Validation | Training before 2024; time-based validation on 2024 |
| Ensemble | Equal-weight combination of tuned and targeted-weather LightGBM models |
| Postprocessing | Global prediction scaling and clipping to each group's capacity |

Lessons recorded during development:

- Wind-vector features need validation rather than assuming that more weather features help.
- Broad aggregation can mix variables with different units and physical meanings; targeted weather features offer a more interpretable alternative.
- Separate models by KPX group and fit missing-value handling on the training split.
- Evaluate both forecast error and the settlement-oriented FiCR component when selecting postprocessing.
- Preserve unsuccessful experiments and their reasoning so later changes have a clear comparison point.

## Repository Guide

- `src/dacon_wind/`: data readers, weather/calendar features, time split, metric and models.
- `scripts/`: training, local evaluation, ensemble/postprocessing and submission validation.
- `notebooks/`: baseline and metric reference notebooks.
- [Experiment log](docs/experiment_log.md) and [experiment feedback](docs/experiment_feedback.md): recorded comparisons and lessons.
- [Workflow](docs/workflow.md), `start-plan.md`, `AGENTS.md` and `prompts/`: development references. Their progress notes are historical development references.

## Local Setup and Data

`requirements.txt` preserves the original pinned environment (UTF-16 encoded), including development dependencies. Create an isolated environment and install it from the project root:

```shell
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r requirements.txt
```

There is no automatic dataset download. Only users authorized under the competition conditions should obtain the official data and place the files locally:

| Local path | Required input |
|---|---|
| `data/raw/train/train_labels.csv` | Group generation labels |
| `data/raw/train/ldaps_train.csv`, `gfs_train.csv` | Training weather forecasts |
| `data/raw/test/ldaps_test.csv`, `gfs_test.csv` | Evaluation weather forecasts |
| `data/raw/sample_submission.csv` | Official submission schema and row order |

The baseline scripts shown below do not require the provided SCADA files. Do not add any official dataset files to Git.

## Recorded Submission Pipeline

Run from the project root with the authorized inputs available. Scripts use fixed experiment IDs and refuse to overwrite existing generated artifacts; use a fresh working directory for reruns.

For the tuned single-model submission:

```shell
python scripts/train_lgbm_tuned_submit.py
python scripts/validate_submission.py submissions/lgbm_003_tuned.csv
```

`python scripts/tune_lgbm_cv.py` runs the earlier parameter search; the submission script already contains the selected parameters.

For the recorded ensemble and scale-1.10 submission, the complete artifact dependency order is:

```shell
python scripts/train_lgbm_tuned_submit.py
python scripts/train_lgbm_targeted_weather_submit.py
python scripts/create_ensemble_submission.py
python scripts/create_ficr_postprocess_submission.py
python scripts/create_scale_110_submission.py
python scripts/validate_submission.py submissions/lgbm_008_scale_110.csv
```

The 1.03 postprocessing step is included because the 1.10 script reads the earlier submission for comparison and alignment. The final 1.10 predictions are calculated directly from the unscaled ensemble, rather than multiplying the two scales together.

These commands recreate the recorded test-prediction pipeline. They do not rerun the full validation search; individual CV scripts document those experiments. Full training and inference require the excluded raw data.

## Limitations

- Local selection used the 2024 validation period. Repeated model/scale selection on that period can overfit; generalization requires validation on additional periods.
- Global scaling can trade forecast accuracy against settlement-oriented behavior. Its effect needs independent validation before reuse.
- Current weather aggregation drops `data_available_kst_dtm` and has no explicit prediction-cutoff filter. The official rules specify a cutoff of the previous day at 14:00 KST. Availability of every input against that cutoff has not been verified from raw data. A fresh competition-valid run needs this check before training.
- `validate_submission.py` checks shape, columns, missing/non-numeric values and prediction bounds. It does not independently verify forecast IDs/timestamps against the sample; ensemble/postprocessing scripts perform those alignment checks.

## Sources and License Status

Official competition, data, rules and metric references are listed in [docs/SOURCES.md](docs/SOURCES.md). Dataset usage conditions apply separately to the code. This repository currently specifies no project-wide reuse license; referenced competition material and dependencies are not relicensed by this documentation.
