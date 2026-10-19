from src.config import FEATURES, TARGET
from src.data import clean


def test_clean_removes_missing_values(raw_df):
    out = clean(raw_df)
    assert out[FEATURES].isna().sum().sum() == 0


def test_clean_binarises_target(raw_df):
    out = clean(raw_df)
    assert "num" not in out.columns
    assert set(out[TARGET].unique()) <= {0, 1}
    assert (out[TARGET] == (raw_df["num"] > 0).astype(int)).all()


def test_clean_keeps_all_features(raw_df):
    out = clean(raw_df)
    assert set(FEATURES) <= set(out.columns)
    assert TARGET in out.columns
    assert len(out) == len(raw_df)


def test_clean_drops_duplicates(raw_df):
    doubled = raw_df.iloc[[0, 0, 1]]
    assert len(clean(doubled)) == 2


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    assert raw_df.equals(before)
