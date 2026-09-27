"""Step 6 - Data preparation: missing data, outliers, categorical features, class imbalance.

Everything here is fitted on the training split only and shipped inside each model's sklearn Pipeline,
so the exact same transformations run at scoring time.

Two branches, because the model families need different things:

                    linear branch (Logistic Regression)             tree branch (RF, HistGB, LightGBM, XGBoost)
  missing           median by part number (train) + NaN flags       kept as NaN, learned natively
  outliers          winsorise at train p0.1 / p99.9, log1p counts   not needed (split-based, rank-invariant)
  scale             standardise                                     not needed
  categorical       one-hot, unseen category -> all zeros           ordinal codes, unseen category -> NaN
  imbalance         no resampling; positive class weight tuned in {1, sqrt(neg/pos)} - see imbalance_weights()
                    and the ablation in src/imbalance_experiment.py

Sentinel glitch codes and single-point spikes were already removed while building features (src/features.py).

Usage:
    python -m src.preprocess     # fits on train and prints what each step did
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import PROCESSED_DIR
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

COUNT_FEATURES = [
    "csi", "csn", "cso", "days_since_install", "related_faults_3d", "related_faults_7d", "related_faults_30d",
    "nuisance_faults_7d", "caution_faults_7d", "troubleshooting_30d", "troubleshooting_install",
    "mel_deferrals_30d", "prior_unscheduled_removals", "prior_nff_removals", "shop_visits",
]
# Only sensor-derived features can be missing; the rest are always computable
MISSING_FLAG_FEATURES = ["primary_z_signed", "primary_slope_signed", "primary_dev_part_signed",
                         "sensor_missing_ratio_7d"]


class PartMedianImputer(BaseEstimator, TransformerMixin):
    """Fill NaN with the training median of the same part number, falling back to the global median.

    A part number never seen in training (a new component type) falls back to the global median.
    Optionally appends a 0/1 missing flag for selected columns so the model knows the value was imputed.
    """

    def __init__(self, group_col: str = "part_no", flag_cols: list[str] | None = None):
        self.group_col = group_col
        self.flag_cols = flag_cols

    def fit(self, X: pd.DataFrame, y=None):
        num = X.drop(columns=[self.group_col])
        self.columns_ = list(num.columns)
        self.global_median_ = num.median()
        self.group_median_ = num.groupby(X[self.group_col].astype(str), observed=True).median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        num = X[self.columns_].copy()
        flags = num[self.flag_cols].isna().astype(float).add_suffix("_missing") if self.flag_cols else None
        by_group = self.group_median_.reindex(X[self.group_col].astype(str).to_numpy())
        by_group.index = num.index
        num = num.fillna(by_group).fillna(self.global_median_)
        return pd.concat([num, flags], axis=1) if flags is not None else num

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_ + [f"{c}_missing" for c in (self.flag_cols or [])])


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to training quantiles so a handful of extreme values cannot dominate a linear model.

    Limits are wide (0.1% / 99.9%) on purpose: the top of the sensor-drift distribution is exactly where the
    failing units are, so we only tame the extreme tail rather than flatten the signal.
    """

    def __init__(self, lower: float = 0.001, upper: float = 0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lo_ = X.quantile(self.lower)
        self.hi_ = X.quantile(self.upper)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        return pd.DataFrame(X).clip(self.lo_, self.hi_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def _log1p_counts(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    cols = [c for c in COUNT_FEATURES if c in X.columns]
    X[cols] = np.log1p(X[cols].clip(lower=0))
    return X


def linear_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", PartMedianImputer(flag_cols=MISSING_FLAG_FEATURES)),
        ("log_counts", FunctionTransformer(_log1p_counts, feature_names_out="one-to-one")),
        ("winsorise", Winsorizer()),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES + ["part_no"]),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def tree_preprocessor() -> ColumnTransformer:
    categorical = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                 encoded_missing_value=np.nan)
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def categorical_indices() -> list[int]:
    """Column positions of the categorical features after tree_preprocessor (they come last)."""
    return list(range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))


def imbalance_weights(y: pd.Series) -> dict:
    """Class-weighting parameters for every model family, computed from the training labels.

    Class weights instead of resampling:
      * SMOTE would interpolate between snapshots of different units and invent sensor histories that
        never happened, and consecutive snapshots are already correlated.
      * undersampling negatives throws away healthy-unit variety (different parts, climates, ages).
      * weighting keeps every real row and only changes how much a missed removal costs during training.
    Validation and test keep the natural ~2% rate so recall and alert rate are measured honestly.
    """
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_weight": n_neg / n_pos,                     # full ratio ("balanced")
        "pos_weight_sqrt": float(np.sqrt(n_neg / n_pos)), # moderate weight
        "class_weight": {0: 1.0, 1: n_neg / n_pos},       # sklearn models
        # The ablation (src/imbalance_experiment.py) showed the full ratio gives no recall gain at the alert
        # budget and badly inflates probabilities, so model tuning searches {1, pos_weight_sqrt} only.
    }


def load_split(name: str, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    if df is None:
        df = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
    mask = {"train": df.split == "train", "valid": df.split == "valid",
            "test_strict": df.test_strict, "test_fleet": df.test_fleet}[name]
    part = df[mask]
    return part[NUMERIC_FEATURES + CATEGORICAL_FEATURES], part.label


def main():
    df = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
    X_tr, y_tr = load_split("train", df)
    X_te, _ = load_split("test_strict", df)

    lin = linear_preprocessor().fit(X_tr, y_tr)
    tree = tree_preprocessor().fit(X_tr, y_tr)
    Z_lin = lin.transform(X_te)
    Z_tree = tree.transform(X_te)

    print("Missing data")
    miss = X_tr[NUMERIC_FEATURES].isna().mean()
    print(f"  train features with NaN: {(miss > 0).sum()} of {len(NUMERIC_FEATURES)} "
          f"(max {miss.max():.1%}: {miss.idxmax()})")
    print(f"  linear branch output NaN: {int(np.isnan(Z_lin).sum())}   "
          f"tree branch keeps NaN: {int(np.isnan(Z_tree[:, :len(NUMERIC_FEATURES)]).sum()):,}")

    w = lin.named_transformers_["num"].named_steps["winsorise"]
    lo, hi = w.lo_, w.hi_
    print("\nOutliers (linear branch, after log1p): winsorise limits for a few features")
    for c in ["primary_z_signed", "primary_slope_signed", "related_faults_7d", "csi"]:
        print(f"  {c:<22} [{lo[c]:.2f}, {hi[c]:.2f}]")

    print(f"\nCategorical: one-hot width {sum(len(c) for c in lin.named_transformers_['cat'].categories_)}"
          f" columns from {len(CATEGORICAL_FEATURES)} features; linear branch output shape {Z_lin.shape}")

    w = imbalance_weights(y_tr)
    print(f"\nClass imbalance: {w['n_pos']:,} positives vs {w['n_neg']:,} negatives "
          f"-> positive class weight {w['pos_weight']:.1f}")

    # a component type the model has never seen, with no sensor data at all
    new = X_te.iloc[[0]].copy()
    new["part_no"] = pd.Categorical(["NEW-PART-01"])
    new[[c for c in NUMERIC_FEATURES if "z" in c or "slope" in c or "mean_7d" in c or "dev" in c]] = np.nan
    print("\nNew part number + no sensor data:")
    print(f"  linear branch -> finite: {np.isfinite(lin.transform(new)).all()}")
    print(f"  tree branch   -> part_no code: {tree.transform(new)[0, len(NUMERIC_FEATURES)]} (NaN = unknown)")


if __name__ == "__main__":
    main()
