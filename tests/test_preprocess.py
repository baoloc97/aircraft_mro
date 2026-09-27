"""Data-preparation guarantees: fitted on train only, robust to new components and missing data."""
import numpy as np
import pandas as pd
import pytest

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.preprocess import (PartMedianImputer, imbalance_weights, linear_preprocessor, load_split,
                            tree_preprocessor)

SENSOR_FEATURES = [c for c in NUMERIC_FEATURES if any(k in c for k in ("_z", "slope", "mean_7d", "dev_part"))]


@pytest.fixture(scope="module")
def splits():
    X_tr, y_tr = load_split("train")
    X_te, _ = load_split("test_strict")
    return X_tr.sample(20_000, random_state=0), y_tr.sample(20_000, random_state=0), X_te


def test_imputer_uses_training_statistics_only(splits):
    X_tr, _, X_te = splits
    imp = PartMedianImputer().fit(X_tr[["primary_z_signed", "part_no"]])
    before = imp.group_median_.copy()
    imp.transform(X_te[["primary_z_signed", "part_no"]])
    pd.testing.assert_frame_equal(before, imp.group_median_)
    expected = X_tr.groupby(X_tr.part_no.astype(str), observed=True).primary_z_signed.median()
    np.testing.assert_allclose(imp.group_median_.primary_z_signed.loc[expected.index], expected)


def test_new_part_number_and_no_sensor_data(splits):
    X_tr, y_tr, X_te = splits
    row = X_te.iloc[[0]].copy()
    row["part_no"] = pd.Categorical(["NEW-PART-01"])
    row["aircraft_type"] = pd.Categorical(["A350-900"])
    row[SENSOR_FEATURES] = np.nan
    lin = linear_preprocessor().fit(X_tr, y_tr).transform(row)
    tree = tree_preprocessor().fit(X_tr, y_tr).transform(row)
    assert np.isfinite(lin).all()
    cat = tree[0, len(NUMERIC_FEATURES):]
    assert np.isnan(cat[CATEGORICAL_FEATURES.index("part_no")])
    assert np.isnan(cat[CATEGORICAL_FEATURES.index("aircraft_type")])


def test_linear_branch_has_no_nan_and_tree_branch_keeps_it(splits):
    X_tr, y_tr, X_te = splits
    assert not np.isnan(linear_preprocessor().fit(X_tr, y_tr).transform(X_te)).any()
    assert np.isnan(tree_preprocessor().fit(X_tr, y_tr).transform(X_te)).any()


def test_winsor_limits_do_not_move_with_scoring_data(splits):
    X_tr, y_tr, X_te = splits
    lin = linear_preprocessor().fit(X_tr, y_tr)
    w = lin.named_transformers_["num"].named_steps["winsorise"]
    hi = w.hi_.copy()
    lin.transform(X_te)
    pd.testing.assert_series_equal(hi, w.hi_)


def test_class_weight_matches_label_ratio(splits):
    _, y_tr, _ = splits
    w = imbalance_weights(y_tr)
    assert w["pos_weight"] == pytest.approx((y_tr == 0).sum() / (y_tr == 1).sum())
