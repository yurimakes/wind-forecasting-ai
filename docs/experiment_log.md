# Experiment Log

Record every meaningful local experiment and DACON submission here. Keep paths relative to the project root.

| exp_id | submit_file | model | features | local_score | public_score | public_1_nmae | public_ficr | notes |
|---|---|---|---|---:|---:|---:|---:|---|
| 001_baseline_rf | submissions/baseline_rf_001.csv | RandomForest | baseline features | - | 0.5879246832 | 0.8637123595 | 0.312137007 | first submitted baseline |
| baseline_rf_001_valid_2024 | outputs/predictions/baseline_rf_001_valid_2024.csv | RandomForest | baseline calendar + LDAPS/GFS mean features | 0.5777342168 | - | - | - | 2024 time-based local validation pipeline |
| lgbm_001 | outputs/predictions/lgbm_001_valid_2024.csv | LightGBM v1 | baseline calendar + LDAPS/GFS mean features | - | - | - | - | 2024 time-based local validation pipeline; run `python scripts/train_lgbm_cv.py` to generate artifacts |
| lgbm_001_submit_2025 | submissions/lgbm_001.csv | LightGBM v1 | baseline calendar + LDAPS/GFS mean features | - | - | - | - | final 2025 test submission pipeline; trains on all available labels and writes artifacts when `python scripts/train_lgbm_submit.py` is run |

## Checklist

- Use time-based validation, never random split.
- Save models to `outputs/models/`.
- Save validation or test predictions to `outputs/predictions/`.
- Save run logs to `outputs/logs/`.
- Save final submission CSVs to `submissions/`.
- Run `python scripts/validate_submission.py submissions/<file>.csv` before upload.


