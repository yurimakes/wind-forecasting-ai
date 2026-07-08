# Experiment Log

Record every meaningful local experiment and DACON submission here. Keep paths relative to the project root.

| exp_id | submit_file | model | features | local_score | public_score | public_1_nmae | public_ficr | notes |
|---|---|---|---|---:|---:|---:|---:|---|
| 001_baseline_rf | submissions/baseline_rf_001.csv | RandomForest | baseline features | - | 0.5879246832 | 0.8637123595 | 0.312137007 | first submitted baseline |
| baseline_rf_001_valid_2024 | outputs/predictions/baseline_rf_001_valid_2024.csv | RandomForest | baseline calendar + LDAPS/GFS mean features | 0.5777342168 | - | - | - | 2024 time-based local validation pipeline |
| lgbm_001 | outputs/predictions/lgbm_001_valid_2024.csv | LightGBM v1 | baseline calendar + LDAPS/GFS mean features | 0.5984976879 | - | - | - | 2024 time-based local validation; one_minus_nmae=0.8656673250, ficr=0.3313280508 |
| lgbm_001_submit_2025 | submissions/lgbm_001.csv | LightGBM v1 | baseline calendar + LDAPS/GFS mean features | - | 0.6024555766 | 0.8659952964 | 0.3389158568 | DACON v2 public score; rank 279 at submission time |
| lgbm_002_wind | outputs/predictions/lgbm_002_wind_valid_2024.csv | LightGBM v1 | baseline calendar + LDAPS/GFS mean features + wind vector derivatives | 0.5966447814 | - | - | - | 2024 local validation; one_minus_nmae=0.8657231002, ficr=0.3275664627; worse than lgbm_001, no submission |


## Checklist

- Use time-based validation, never random split.
- Save models to `outputs/models/`.
- Save validation or test predictions to `outputs/predictions/`.
- Save run logs to `outputs/logs/`.
- Save final submission CSVs to `submissions/`.
- Run `python scripts/validate_submission.py submissions/<file>.csv` before upload.
