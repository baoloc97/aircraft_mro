"""Leakage test: features computed on the full history must equal features computed on data cut at day t.

If a feature changes when future rows are removed, it is using information from after the snapshot.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DIR
from src.features import FEATURES, build_features, load_raw, raw_as_of

CUT_DATES = ["2024-06-15", "2025-02-10", "2025-09-01"]


@pytest.fixture(scope="module")
def data():
    snaps = pd.read_parquet(PROCESSED_DIR / "snapshots.parquet")
    return snaps, load_raw()


@pytest.mark.parametrize("cut_date", CUT_DATES)
def test_features_do_not_use_future_data(data, cut_date):
    snaps, raw = data
    t = pd.Timestamp(cut_date)
    at_t = snaps[snaps.snapshot_date == snaps.snapshot_date[snaps.snapshot_date <= t].max()].reset_index(drop=True)
    full = build_features(at_t, dict(raw)).sort_values("installation_id").reset_index(drop=True)
    past = build_features(at_t, raw_as_of(raw, at_t.snapshot_date.iloc[0])).sort_values("installation_id").reset_index(drop=True)

    leaking = []
    for c in FEATURES:
        a, b = full[c], past[c]
        if isinstance(a.dtype, pd.CategoricalDtype):
            same = (a.astype(str) == b.astype(str)).all()
        else:
            same = np.allclose(a.astype(float), b.astype(float), equal_nan=True, rtol=1e-9, atol=1e-9)
        if not same:
            leaking.append(c)
    assert not leaking, f"features that change when future data is removed: {leaking}"
