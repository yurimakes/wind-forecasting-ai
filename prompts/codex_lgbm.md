Build a minimal LightGBM experiment for the DACON wind forecasting project.

Rules:
- Use a time-based validation split. Do not use random validation split.
- Fix random seeds and make train/inference reproducible.
- Do not use remote model APIs inside train or inference.
- Do not modify `data/raw`.
- Do not overwrite existing submissions unless explicitly asked.
- Save models to `outputs/models/`.
- Save predictions to `outputs/predictions/`.
- Save logs to `outputs/logs/`.
- Save submission CSVs to `submissions/` with a unique experiment id.
- Validate any submission candidate with:

```powershell
python scripts/validate_submission.py submissions/<file>.csv
```

Use the official local metric from `src/dacon_wind/metric.py` and record results in `docs/experiment_log.md`.
