from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TrainData:
    labels: pd.DataFrame
    ldaps: pd.DataFrame
    gfs: pd.DataFrame


@dataclass(frozen=True)
class TestData:
    sample_submission: pd.DataFrame
    ldaps: pd.DataFrame
    gfs: pd.DataFrame


def load_train_data(data_dir: Path | str = Path("data/raw/train")) -> TrainData:
    """Load the raw train files used by the baseline RandomForest pipeline."""
    data_dir = Path(data_dir)

    labels = pd.read_csv(data_dir / "train_labels.csv", encoding="utf-8-sig")
    ldaps = pd.read_csv(data_dir / "ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(data_dir / "gfs_train.csv", encoding="utf-8-sig")

    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    ldaps["forecast_kst_dtm"] = pd.to_datetime(ldaps["forecast_kst_dtm"])
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])

    return TrainData(labels=labels, ldaps=ldaps, gfs=gfs)


def load_test_data(
    test_dir: Path | str = Path("data/raw/test"),
    sample_path: Path | str = Path("data/raw/sample_submission.csv"),
) -> TestData:
    """Load the raw test files and sample submission for final inference."""
    test_dir = Path(test_dir)
    sample_path = Path(sample_path)

    sample_submission = pd.read_csv(sample_path, encoding="utf-8-sig")
    ldaps = pd.read_csv(test_dir / "ldaps_test.csv", encoding="utf-8-sig")
    gfs = pd.read_csv(test_dir / "gfs_test.csv", encoding="utf-8-sig")

    sample_submission["forecast_kst_dtm"] = pd.to_datetime(
        sample_submission["forecast_kst_dtm"]
    )
    ldaps["forecast_kst_dtm"] = pd.to_datetime(ldaps["forecast_kst_dtm"])
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])

    return TestData(sample_submission=sample_submission, ldaps=ldaps, gfs=gfs)
