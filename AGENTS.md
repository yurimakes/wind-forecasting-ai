# Project Rules

Before editing data, feature, metric, train, inference, or submission code, read `docs/competition_agent_context.md` first.

## Competition Safety Rules

- Do not modify files under `data/raw`.
- Do not modify existing files under `submissions` unless explicitly asked.
- Do not use random validation splits.
- Use time-based validation for local experiments.
- Do not use future information after the prediction cutoff time.
- Do not use remote model APIs inside train or inference code.
- Train and inference code must be separated and reproducible.
- Fix random seeds, record configs, and save generated artifacts with clear experiment ids.
- Any external data must have source, collection time, license, and reproducibility notes.

## Experiment Hygiene

- Save submission CSVs under `submissions/` with a unique experiment id.
- Save model artifacts under `outputs/models/`.
- Save prediction artifacts under `outputs/predictions/`.
- Save logs and run summaries under `outputs/logs/`.
- Validate every submission candidate with `scripts/validate_submission.py` before upload.
- Record important experiment results in `docs/experiment_log.md`.

## Current Baseline Direction

- Main baseline family: LightGBM time-based CV.
- Optimize both local total score and leaderboard robustness.
- Do not overfit to public leaderboard; keep validation logic time-aware.

## Current Best Baseline

- As of 2026-07-09 KST, `lgbm_003_tuned_submit` is the best submitted baseline.
- Submission file: `submissions/lgbm_003_tuned.csv`.
- Local validation total_score: 0.6033279875.
- DACON public total_score: 0.60516.
- Use this run as the comparison point for new model families and ensemble candidates.

## Documentation Hygiene

- Keep `README.md` and `docs/experiment_log.md` synchronized after meaningful experiments or submissions.
- Preserve older experiment rows unless they are clearly duplicated or incorrect.
- Document generated artifacts by relative path, but do not commit ignored raw data, model files, predictions, logs, or submission outputs unless explicitly requested.
