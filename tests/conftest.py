import numpy as np
import pandas as pd
import pytest

from src.config import COLUMNS


@pytest.fixture
def raw_df():
    """Small synthetic frame shaped like the raw UCI file, with gaps."""
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(95, 200, n),
        "chol": rng.integers(130, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 5, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
        "num": rng.integers(0, 5, n),
    })[COLUMNS]
    df.loc[[1, 5], "ca"] = np.nan
    df.loc[[2], "thal"] = np.nan
    return df


@pytest.fixture
def sample_patient():
    return {"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 2, "thalach": 150, "exang": 1,
            "oldpeak": 2.3, "slope": 3, "ca": 2, "thal": 7}
