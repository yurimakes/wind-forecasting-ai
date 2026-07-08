from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {
    "kpx_group_1": 21600,
    "kpx_group_2": 21600,
    "kpx_group_3": 21000,
}
DEFAULT_SAMPLE_PATH = Path("data/raw/sample_submission.csv")


def validate_submission(submission_path: Path, sample_path: Path = DEFAULT_SAMPLE_PATH) -> list[str]:
    errors: list[str] = []

    if not sample_path.exists():
        return [f"Sample submission not found: {sample_path}"]
    if not submission_path.exists():
        return [f"Submission not found: {submission_path}"]

    sample_df = pd.read_csv(sample_path)
    submission_df = pd.read_csv(submission_path)

    if submission_df.shape != sample_df.shape:
        errors.append(
            f"Shape mismatch: expected {sample_df.shape}, got {submission_df.shape}"
        )

    expected_cols = list(sample_df.columns)
    actual_cols = list(submission_df.columns)
    if actual_cols != expected_cols:
        errors.append(f"Column mismatch: expected {expected_cols}, got {actual_cols}")

    if submission_df.isna().any().any():
        errors.append("Submission contains NaN values")

    for col in TARGET_COLS:
        if col not in submission_df.columns:
            continue

        values = pd.to_numeric(submission_df[col], errors="coerce")
        if values.isna().any():
            errors.append(f"{col} contains non-numeric values")
            continue

        if (values < 0).any():
            errors.append(f"{col} contains negative predictions")

        capacity = CAPACITY_KWH[col]
        if (values > capacity).any():
            errors.append(f"{col} exceeds capacity upper bound {capacity}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a DACON wind forecasting submission CSV."
    )
    parser.add_argument("submission", type=Path, help="Path to submission CSV")
    parser.add_argument(
        "--sample",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
        help="Path to sample_submission.csv",
    )
    args = parser.parse_args()

    errors = validate_submission(args.submission, args.sample)
    if errors:
        print("Submission validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Submission validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
