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

## Constraints

- Do not modify `data/raw`.
- Do not modify existing submission files unless explicitly asked.
- Do not use remote model APIs inside train or inference code.
