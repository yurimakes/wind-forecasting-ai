# Experiment Log

Record every meaningful local experiment and DACON submission here. Keep paths relative to the project root.

| exp_id | submit_file | model | features | local_score | public_score | public_1_nmae | public_ficr | notes |
|---|---|---|---|---:|---:|---:|---:|---|
| 001_baseline_rf | submissions/baseline_rf_001.csv | RandomForest | baseline features | - | 0.5879246832 | 0.8637123595 | 0.312137007 | first submitted baseline |

## Checklist

- Use time-based validation, never random split.
- Save models to `outputs/models/`.
- Save validation or test predictions to `outputs/predictions/`.
- Save run logs to `outputs/logs/`.
- Save final submission CSVs to `submissions/`.
- Run `python scripts/validate_submission.py submissions/<file>.csv` before upload.
