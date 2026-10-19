"""Download and clean the UCI Heart Disease (Cleveland) dataset.

Usage:  python -m src.data
"""
import logging

import numpy as np
import pandas as pd
import requests

from src.config import (
    CLEAN_FILE, COLUMNS, DATA_URL, FEATURES, RAW_FILE, TARGET,
)

log = logging.getLogger(__name__)


def download(url: str = DATA_URL, dest=RAW_FILE, force: bool = False):
    """Download the raw file if it is not already present."""
    if dest.exists() and not force:
        log.info("Raw data already present at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_raw(path=RAW_FILE) -> pd.DataFrame:
    """Read the headerless raw file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, impute missing values and binarise the target.

    `num` is 0 (no disease) .. 4 (severe); anything > 0 means disease present.
    """
    df = df.copy()
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates()
    df = df.dropna(subset=["num"])

    # Only `ca` and `thal` have gaps (6 rows total); impute with the mode
    # since both are discrete codes.
    for col in FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    int_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "slope", "ca", "thal"]
    df[int_cols] = df[int_cols].astype(np.int64)
    return df.reset_index(drop=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download()
    df = clean(load_raw())
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)
    log.info("Saved %d rows to %s (positive rate %.2f)",
             len(df), CLEAN_FILE, df[TARGET].mean())


if __name__ == "__main__":
    main()
