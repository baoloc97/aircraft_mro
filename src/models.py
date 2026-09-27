"""Model catalogue: every candidate model is declared once here.

Each entry says which preprocessing branch it needs, how to build the estimator for given hyper-parameters and
class weight, which extra fit arguments it takes, and its search space. Training, the imbalance ablation and the
scorer all build models through this registry, so adding a model means adding one entry, not editing callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import SEED
from src.preprocess import categorical_indices, linear_preprocessor, tree_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    make_estimator: Callable[[dict, float, int], object]     # (params, positive-class weight, seed) -> estimator
    make_preprocessor: Callable[[], object]
    search_space: dict
    n_iter: int
    fit_params: Callable[[], dict] = field(default=lambda: {})


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "LogisticRegression": ModelSpec(
        make_estimator=lambda p, w, seed: LogisticRegression(max_iter=3000, class_weight={0: 1.0, 1: w}, **p),
        make_preprocessor=linear_preprocessor,
        search_space={"C": [0.01, 0.03, 0.1, 0.3, 1.0], "weighted": [False, True]},
        n_iter=10,
    ),
    "RandomForest": ModelSpec(
        make_estimator=lambda p, w, seed: RandomForestClassifier(
            n_estimators=300, class_weight={0: 1.0, 1: w}, n_jobs=-1, random_state=seed, **p),
        make_preprocessor=tree_preprocessor,
        search_space={"max_depth": [8, 12, 16, None], "min_samples_leaf": [10, 30, 80],
                      "max_features": ["sqrt", 0.3, 0.5], "weighted": [False, True]},
        n_iter=8,
    ),
    "HistGradientBoosting": ModelSpec(
        make_estimator=lambda p, w, seed: HistGradientBoostingClassifier(
            categorical_features=categorical_indices(), class_weight={0: 1.0, 1: w}, random_state=seed, **p),
        make_preprocessor=tree_preprocessor,
        search_space={"learning_rate": [0.03, 0.05, 0.1], "max_iter": [200, 400, 600],
                      "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [20, 50, 100],
                      "l2_regularization": [0.0, 1.0, 5.0], "weighted": [False, True]},
        n_iter=12,
    ),
    "LightGBM": ModelSpec(
        make_estimator=lambda p, w, seed: lgb.LGBMClassifier(
            scale_pos_weight=w, subsample_freq=1, random_state=seed, verbose=-1, n_jobs=-1, **p),
        make_preprocessor=tree_preprocessor,
        search_space={"learning_rate": [0.02, 0.03, 0.05], "n_estimators": [300, 500, 800],
                      "num_leaves": [15, 31, 63], "min_child_samples": [20, 50, 100],
                      "colsample_bytree": [0.6, 0.8, 1.0], "subsample": [0.7, 0.85, 1.0],
                      "reg_lambda": [0.0, 1.0, 5.0], "weighted": [False, True]},
        n_iter=14,
        fit_params=lambda: {"model__categorical_feature": categorical_indices()},
    ),
    "XGBoost": ModelSpec(
        make_estimator=lambda p, w, seed: xgb.XGBClassifier(
            scale_pos_weight=w, tree_method="hist", random_state=seed, n_jobs=-1, eval_metric="aucpr", **p),
        make_preprocessor=tree_preprocessor,
        search_space={"learning_rate": [0.02, 0.03, 0.05], "n_estimators": [300, 500, 800],
                      "max_depth": [4, 6, 8], "min_child_weight": [1, 5, 10],
                      "colsample_bytree": [0.6, 0.8, 1.0], "subsample": [0.7, 0.85, 1.0],
                      "reg_lambda": [0.0, 1.0, 5.0], "weighted": [False, True]},
        n_iter=14,
    ),
}


def build_model(name: str, params: dict, pos_weight: float, seed: int = SEED) -> Pipeline:
    """Preprocessing + estimator. `params["weighted"]` switches the positive-class weight on (else weight 1)."""
    spec = MODEL_REGISTRY[name]
    p = dict(params)
    w = pos_weight if p.pop("weighted", False) else 1.0
    return Pipeline([("prep", spec.make_preprocessor()), ("model", spec.make_estimator(p, w, seed))])


def fit_params(name: str) -> dict:
    spec = MODEL_REGISTRY.get(name)
    return spec.fit_params() if spec else {}


class RuleBaseline:
    """Score = a single engineering rule. No fitting; exists so ML has something honest to beat."""

    RULES = {
        # "replace what is oldest relative to its MTBUR"
        "age": lambda X: X.csi_over_mtbur.to_numpy(),
        # "watch repeat defects": related fault codes in the last 7 days, age as tie-breaker
        "faults": lambda X: X.related_faults_7d.to_numpy() + 0.01 * np.clip(X.csi_over_mtbur.to_numpy(), 0, 5),
    }

    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, X, y=None, **_):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        s = np.nan_to_num(self.RULES[self.kind](X))
        s = s / (s.max() + 1e-9)
        return np.column_stack([1 - s, s])
