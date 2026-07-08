from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dacon_wind.metric import CAPACITY_KWH, TARGET_COLS
from validate_submission import validate_submission


EXP_ID = "ens_001_simple_avg_submit"
RUN_ID = "ens_001_simple_avg_submit"
SUBMISSION_NAME = "ens_001_simple_avg.csv"
BASELINE_PREDICTION = ROOT / "outputs" / "predictions" / "lgbm_003_tuned_test.csv"
TARGETED_PREDICTION = (
    ROOT / "outputs" / "predictions" / "lgbm_005_targeted_weather_test.csv"
)
SAMPLE_SUBMISSION = ROOT / "data" / "raw" / "sample_submission.csv"
VALIDATION_REFERENCE = {
    "exp_id": "ens_001_simple_avg",
    "run_id": "ens_001_simple_avg_valid_2024",
    "selected_candidate": "ens_001_lgbm003_lgbm005_avg",
    "total_score": 0.6037465036,
    "one_minus_nmae": 0.8680552297,
    "ficr": 0.3394377775,
    "weights": {
        "lgbm_003_tuned": 0.5,
        "lgbm_005_targeted_weather": 0.5,
    },
}


def ensure_new_artifacts(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing artifact(s):\n"
            f"{formatted}\n"
            "Remove or rename them before rerunning this script."
        )


def load_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path.relative_to(ROOT)}")

    prediction = pd.read_csv(path, encoding="utf-8-sig")
    required_cols = ["forecast_id", "forecast_kst_dtm", *TARGET_COLS]
    missing = [col for col in required_cols if col not in prediction.columns]
    if missing:
        raise ValueError(f"{path.relative_to(ROOT)} is missing columns: {missing}")

    prediction = prediction.loc[:, required_cols].copy()
    prediction["forecast_kst_dtm"] = pd.to_datetime(prediction["forecast_kst_dtm"])
    return prediction


def assert_aligned(name: str, prediction: pd.DataFrame, sample: pd.DataFrame) -> None:
    if len(prediction) != len(sample):
        raise ValueError(
            f"{name} row count {len(prediction)} does not match sample row count {len(sample)}."
        )
    if not prediction["forecast_id"].equals(sample["forecast_id"]):
        raise ValueError(f"{name} forecast_id alignment mismatch.")
    if not prediction["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"]):
        raise ValueError(f"{name} forecast_kst_dtm alignment mismatch.")


def main() -> None:
    submission_dir = ROOT / "submissions"
    prediction_dir = ROOT / "outputs" / "predictions"
    log_dir = ROOT / "outputs" / "logs"
    for path in (submission_dir, prediction_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    ensemble_prediction_path = prediction_dir / "ens_001_simple_avg_test.csv"
    submission_path = submission_dir / SUBMISSION_NAME
    log_path = log_dir / "ens_001_simple_avg_submit.json"
    ensure_new_artifacts([ensemble_prediction_path, submission_path, log_path])

    sample = pd.read_csv(SAMPLE_SUBMISSION, encoding="utf-8-sig")
    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])

    baseline = load_prediction(BASELINE_PREDICTION)
    targeted = load_prediction(TARGETED_PREDICTION)
    assert_aligned("lgbm_003_tuned_test", baseline, sample)
    assert_aligned("lgbm_005_targeted_weather_test", targeted, sample)

    ensemble = sample.copy()
    for target in TARGET_COLS:
        averaged = (
            0.5 * baseline[target].to_numpy(dtype=float)
            + 0.5 * targeted[target].to_numpy(dtype=float)
        )
        ensemble[target] = np.clip(averaged, 0.0, CAPACITY_KWH[target])

    sample_columns = list(sample.columns)
    ensemble = ensemble.loc[:, sample_columns]
    ensemble["forecast_kst_dtm"] = pd.to_datetime(
        ensemble["forecast_kst_dtm"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    ensemble.to_csv(ensemble_prediction_path, index=False, encoding="utf-8-sig")
    ensemble.to_csv(submission_path, index=False, encoding="utf-8-sig")

    validation_errors = validate_submission(submission_path, SAMPLE_SUBMISSION)
    if validation_errors:
        raise ValueError(
            "Generated submission failed validation:\n"
            + "\n".join(f"- {error}" for error in validation_errors)
        )

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Create DACON submission from the best validation ensemble.",
        "input_predictions": {
            "lgbm_003_tuned": str(BASELINE_PREDICTION.relative_to(ROOT)),
            "lgbm_005_targeted_weather": str(TARGETED_PREDICTION.relative_to(ROOT)),
        },
        "weights": VALIDATION_REFERENCE["weights"],
        "alignment": {
            "sample_submission": str(SAMPLE_SUBMISSION.relative_to(ROOT)),
            "keys": ["forecast_id", "forecast_kst_dtm"],
            "status": "both member prediction files match sample_submission order",
        },
        "postprocessing": {
            "prediction_clipping": "Per target group, clipped to [0, group_capacity].",
            "submission_columns": sample_columns,
        },
        "artifacts": {
            "test_predictions": str(ensemble_prediction_path.relative_to(ROOT)),
            "submission": str(submission_path.relative_to(ROOT)),
            "run_summary": str(log_path.relative_to(ROOT)),
        },
        "validation": {
            "script": "scripts/validate_submission.py",
            "passed": True,
        },
        "validation_score_reference": VALIDATION_REFERENCE,
        "public_score": "pending",
    }

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved ensemble test predictions: {ensemble_prediction_path.relative_to(ROOT)}")
    print(f"Saved submission: {submission_path.relative_to(ROOT)}")
    print(f"Saved run summary: {log_path.relative_to(ROOT)}")
    print("Submission validation passed.")


if __name__ == "__main__":
    main()
