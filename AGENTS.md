# Project Rules

- Do not modify files under `data/raw`.
- Do not modify existing files under `submissions` unless explicitly asked.
- Do not use random validation splits.
- Use time-based validation for local experiments.
- Train and inference code must be reproducible: fix random seeds, record configs, and save generated artifacts with clear experiment ids.
- Do not use remote model APIs inside train or inference code.

# Experiment Hygiene

- Save submission CSVs under `submissions/` with a unique experiment id.
- Save model artifacts under `outputs/models/`.
- Save prediction artifacts under `outputs/predictions/`.
- Save logs and run summaries under `outputs/logs/`.
- Validate every submission candidate with `scripts/validate_submission.py` before upload.

# 
- Before editing data, feature, metric, train, inference, or submission code, read docs/competition_agent_context.md first.