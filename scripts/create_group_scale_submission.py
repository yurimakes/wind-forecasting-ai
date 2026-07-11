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


EXP_ID = "ens_004_group_scale_submit"
RUN_ID = "ens_004_group_scale_submit"
SUBMISSION_NAME = "ens_004_group_scale.csv"
GROUP_SCALES = {
    "kpx_group_1": 1.110,
    "kpx_group_2": 1.020,
    "kpx_group_3": 1.130,
}

SOURCE_TEST_PREDICTION = ROOT / "outputs" / "predictions" / "ens_001_simple_avg_test.csv"
CURRENT_BEST_PUBLIC_SUBMISSION = ROOT / "submissions" / "lgbm_008_scale_110.csv"
SAMPLE_SUBMISSION = ROOT / "data" / "raw" / "sample_submission.csv"
OUTPUT_TEST_PREDICTION = ROOT / "outputs" / "predictions" / "ens_004_group_scale_test.csv"
OUTPUT_SUBMISSION = ROOT / "submissions" / SUBMISSION_NAME
OUTPUT_LOG = ROOT / "outputs" / "logs" / "ens_004_group_scale_submit.json"

VALIDATION_REFERENCE = {
    "experiment": "ens_004_group_scale_cv",
    "source_validation_prediction": (
        "outputs/predictions/ens_001_simple_avg_valid_2024_best.csv"
    ),
    "selected_group_scales": GROUP_SCALES,
    "operation": (
        "multiply each target prediction by its selected group-wise scale, "
        "then clip to each group capacity"
    ),
    "total_score": 0.6248401656,
    "one_minus_nmae": 0.8696017636,
    "ficr": 0.3800785675,
    "comparison_versus_lgbm_008_validation_global_1_10": {
        "reference_experiment": "lgbm_008_scale_upper_sweep_cv",
        "reference_total_score": 0.6184267418,
        "reference_one_minus_nmae": 0.8671491370,
        "reference_ficr": 0.3697043467,
        "delta_total_score": 0.0064134238,
        "delta_one_minus_nmae": 0.0024526266,
        "delta_ficr": 0.0103742208,
    },
    "decision": "strong future submission candidate",
}

CURRENT_BEST_PUBLIC_REFERENCE = {
    "experiment": "lgbm_008_scale_110_submit",
    "submission": "submissions/lgbm_008_scale_110.csv",
    "method": (
        "global scale 1.10 applied to outputs/predictions/ens_001_simple_avg_test.csv, "
        "then capacity clipped"
    ),
    "public_total_score": 0.6211026154,
    "public_one_minus_nmae": 0.8634833072,
    "public_ficr": 0.3787219236,
    "final_ranking_note": "Final ranking still depends on the private leaderboard.",
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
    assert_aligned("lgbm_008_scale_110.csv", current_best_public, sample)

    processed = sample.copy()
    lower_clip_counts: dict[str, int] = {}
    upper_clip_counts: dict[str, int] = {}
    total_clip_counts: dict[str, int] = {}
    for target in TARGET_COLS:
        scaled = source[target].astype(float) * GROUP_SCALES[target]
        clipped = scaled.clip(lower=0.0, upper=CAPACITY_KWH[target])
        processed[target] = clipped
        lower_clip_counts[target] = int((scaled < 0.0).sum())
        upper_clip_counts[target] = int((scaled > CAPACITY_KWH[target]).sum())
        total_clip_counts[target] = int((scaled != clipped).sum())

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
            "Apply the best validation group-wise scales from ens_004_group_scale_cv "
            "to the current best ensemble test prediction. No models are trained."
        ),
        "source_prediction_path": str(SOURCE_TEST_PREDICTION.relative_to(ROOT)),
        "group_scales": GROUP_SCALES,
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
        "capacity_clipping_information": {
            "prediction_clipping": "Per target group, clipped to [0, group_capacity].",
            "capacity_kwh": {target: CAPACITY_KWH[target] for target in TARGET_COLS},
            "lower_clip_counts_after_scaling": lower_clip_counts,
            "upper_clip_counts_after_scaling": upper_clip_counts,
            "total_clip_counts_after_scaling": total_clip_counts,
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
            "Group-wise scaling was selected on 2024 time-based validation and is a "
            "strong candidate, but final ranking depends on the private leaderboard. "
            "Avoid overfitting model selection to repeated public leaderboard feedback."
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
