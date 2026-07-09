from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dacon_wind.metric import CAPACITY_KWH, TARGET_COLS
from validate_submission import validate_submission


EXP_ID = "lgbm_008_scale_110_submit"
RUN_ID = "lgbm_008_scale_110_submit"
SUBMISSION_NAME = "lgbm_008_scale_110.csv"
SCALE = 1.10

SOURCE_TEST_PREDICTION = ROOT / "outputs" / "predictions" / "ens_001_simple_avg_test.csv"
CURRENT_BEST_PUBLIC_SUBMISSION = ROOT / "submissions" / "lgbm_006_ficr_focus.csv"
SAMPLE_SUBMISSION = ROOT / "data" / "raw" / "sample_submission.csv"
OUTPUT_TEST_PREDICTION = ROOT / "outputs" / "predictions" / "lgbm_008_scale_110_test.csv"
OUTPUT_SUBMISSION = ROOT / "submissions" / SUBMISSION_NAME
OUTPUT_LOG = ROOT / "outputs" / "logs" / "lgbm_008_scale_110_submit.json"

VALIDATION_REFERENCE = {
    "experiment": "lgbm_008_scale_upper_sweep_cv",
    "source_validation_prediction": (
        "outputs/predictions/ens_001_simple_avg_valid_2024_best.csv"
    ),
    "selected_scale": SCALE,
    "operation": "multiply all target predictions by 1.10, then clip to each group capacity",
    "total_score": 0.6184267418,
    "one_minus_nmae": 0.8671491370,
    "ficr": 0.3697043467,
    "comparison": (
        "Scale 1.10 beat lgbm_007 scale 1.08 on both total_score and FICR, "
        "and was the peak in the 1.075-1.16 upper sweep, not the upper boundary."
    ),
}

CURRENT_BEST_PUBLIC_REFERENCE = {
    "experiment": "lgbm_006_ficr_focus_submit",
    "submission": "submissions/lgbm_006_ficr_focus.csv",
    "method": (
        "scale 1.03 applied to outputs/predictions/ens_001_simple_avg_test.csv, "
        "then capacity clipped"
    ),
    "public_total_score": 0.6158048399,
    "public_one_minus_nmae": 0.8679909923,
    "public_ficr": 0.3636186875,
    "public_rank_at_submission_time": 245,
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


def load_sample() -> pd.DataFrame:
    if not SAMPLE_SUBMISSION.exists():
        raise FileNotFoundError(
            f"Missing sample submission: {SAMPLE_SUBMISSION.relative_to(ROOT)}"
        )

    sample = pd.read_csv(SAMPLE_SUBMISSION, encoding="utf-8-sig")
    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])
    return sample


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


def summarize_targets(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        target: {
            "min": float(pd.to_numeric(df[target]).min()),
            "max": float(pd.to_numeric(df[target]).max()),
            "mean": float(pd.to_numeric(df[target]).mean()),
        }
        for target in TARGET_COLS
    }


def main() -> None:
    for path in (OUTPUT_SUBMISSION.parent, OUTPUT_TEST_PREDICTION.parent, OUTPUT_LOG.parent):
        path.mkdir(parents=True, exist_ok=True)

    ensure_new_artifacts([OUTPUT_TEST_PREDICTION, OUTPUT_SUBMISSION, OUTPUT_LOG])

    sample = load_sample()
    source = load_prediction(SOURCE_TEST_PREDICTION)
    current_best_public = load_prediction(CURRENT_BEST_PUBLIC_SUBMISSION)

    assert_aligned("ens_001_simple_avg_test", source, sample)
    assert_aligned("lgbm_006_ficr_focus.csv", current_best_public, sample)

    processed = sample.copy()
    clip_counts: dict[str, int] = {}
    lower_clip_counts: dict[str, int] = {}
    upper_clip_counts: dict[str, int] = {}
    for target in TARGET_COLS:
        scaled = source[target].astype(float) * SCALE
        clipped = scaled.clip(lower=0.0, upper=CAPACITY_KWH[target])
        processed[target] = clipped
        lower_clip_counts[target] = int((scaled < 0.0).sum())
        upper_clip_counts[target] = int((scaled > CAPACITY_KWH[target]).sum())
        clip_counts[target] = int((scaled != clipped).sum())

    processed = processed.loc[:, list(sample.columns)]
    processed["forecast_kst_dtm"] = pd.to_datetime(
        processed["forecast_kst_dtm"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    processed.to_csv(OUTPUT_TEST_PREDICTION, index=False, encoding="utf-8-sig")
    processed.to_csv(OUTPUT_SUBMISSION, index=False, encoding="utf-8-sig")

    validation_errors = validate_submission(OUTPUT_SUBMISSION, SAMPLE_SUBMISSION)
    if validation_errors:
        raise ValueError(
            "Generated submission failed validation:\n"
            + "\n".join(f"- {error}" for error in validation_errors)
        )

    summary = {
        "exp_id": EXP_ID,
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Apply the best validation scale from lgbm_008_scale_upper_sweep_cv "
            "to the current best ensemble test prediction. No models are trained."
        ),
        "input_predictions": {
            "source_test_prediction": str(SOURCE_TEST_PREDICTION.relative_to(ROOT)),
            "current_best_public_submission": str(
                CURRENT_BEST_PUBLIC_SUBMISSION.relative_to(ROOT)
            ),
        },
        "scale": SCALE,
        "validation_reference": VALIDATION_REFERENCE,
        "current_best_public_reference": CURRENT_BEST_PUBLIC_REFERENCE,
        "alignment": {
            "sample_submission": str(SAMPLE_SUBMISSION.relative_to(ROOT)),
            "keys": ["forecast_id", "forecast_kst_dtm"],
            "status": (
                "source test prediction and current best public submission match "
                "sample_submission order"
            ),
            "rows": int(len(sample)),
        },
        "capacity_clipping": {
            "prediction_clipping": "Per target group, clipped to [0, group_capacity].",
            "capacity_kwh": {target: CAPACITY_KWH[target] for target in TARGET_COLS},
            "lower_clip_counts_after_scaling": lower_clip_counts,
            "upper_clip_counts_after_scaling": upper_clip_counts,
            "total_clip_counts_after_scaling": clip_counts,
        },
        "target_summary": {
            "source_before_scaling": summarize_targets(source),
            "output_after_scaling_and_clipping": summarize_targets(processed),
        },
        "artifacts": {
            "test_predictions": str(OUTPUT_TEST_PREDICTION.relative_to(ROOT)),
            "submission": str(OUTPUT_SUBMISSION.relative_to(ROOT)),
            "run_summary": str(OUTPUT_LOG.relative_to(ROOT)),
        },
        "validation_result": {
            "script": "scripts/validate_submission.py",
            "submission": str(OUTPUT_SUBMISSION.relative_to(ROOT)),
            "passed": True,
            "errors": [],
        },
        "public_score": "pending",
        "caution": (
            "Scale 1.10 is more aggressive than the public-proven 1.03 scale. "
            "The public leaderboard is only a split of the final evaluation; avoid "
            "private leaderboard risk and excessive public probing."
        ),
    }

    with OUTPUT_LOG.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved postprocessed test predictions: {OUTPUT_TEST_PREDICTION.relative_to(ROOT)}")
    print(f"Saved submission: {OUTPUT_SUBMISSION.relative_to(ROOT)}")
    print(f"Saved run summary: {OUTPUT_LOG.relative_to(ROOT)}")
    print("Submission validation passed.")


if __name__ == "__main__":
    main()
