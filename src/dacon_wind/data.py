from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TrainData:
    labels: pd.DataFrame
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
