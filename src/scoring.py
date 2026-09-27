"""Production scorer: one object that turns feature rows into a calibrated risk score and a risk level.

Used by evaluation, explanation, batch prediction and the API, so every consumer applies the same model,
calibration and thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass
class RiskScorer:
    models: dict                       # name -> fitted sklearn Pipeline
    method: str = "single"             # "single" | "rank" (mean percentile rank across models)
    reference: dict = field(default_factory=dict)   # name -> sorted validation scores (for percentile ranks)
    calibrator: IsotonicRegression | None = None
    threshold: float = np.nan          # raw score at which a component is flagged
    high_threshold: float = np.nan     # raw score for the HIGH tier
    max_alert_rate: float = 0.05       # hard cap per scoring run
    version: str = "v1"
    metadata: dict = field(default_factory=dict)

    # -------------------------------------------------------------- construction
    @classmethod
    def from_decision(cls, decision: str, ranked_models: list[str], models_dir: Path) -> "RiskScorer":
        """Build the scorer from a Step 7 decision string.

        "single: XGBoost" -> one model; "rank vote: top 3" / "rank vote: all 5" -> the best-ranked models;
        "rank vote: A + B + C" -> those models. Rank voting averages each model's percentile rank.
        """
        load = lambda n: joblib.load(models_dir / f"{n}.joblib")["model"]
        kind, _, spec = decision.partition(":")
        spec = spec.strip()
        if kind == "single":
            return cls(models={spec: load(spec)}, method="single", metadata={"decision": decision})
        if spec == "top 3":
            names = ranked_models[:3]
        elif spec == "all 5":
            names = ranked_models
        else:
            names = [n.strip().replace("LR", "LogisticRegression") for n in spec.split("+")]
        return cls(models={n: load(n) for n in names}, method="rank", metadata={"decision": decision})

    # -------------------------------------------------------------- raw score
    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        X = X[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        if self.method == "single":
            return next(iter(self.models.values())).predict_proba(X)[:, 1]
        pct = [np.searchsorted(self.reference[n], m.predict_proba(X)[:, 1], side="right") / len(self.reference[n])
               for n, m in self.models.items()]
        return np.mean(pct, axis=0)

    # -------------------------------------------------------------- calibrated risk
    def fit_calibration(self, raw: np.ndarray, y: np.ndarray):
        # bounded away from 0 and 1: a finite validation set cannot justify certainty either way
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0005, y_max=0.99).fit(raw, y)
        return self

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        """Probability of an unscheduled removal within 30 FC, for display. Isotonic output is a step
        function (many ties), so alerts are decided on the raw score and this value is only reported."""
        return self.calibrator.predict(raw) if self.calibrator is not None else raw

    # -------------------------------------------------------------- alerts
    def flag(self, raw: np.ndarray, run_ids: pd.Series | np.ndarray) -> np.ndarray:
        """Flag raw >= threshold, never more than max_alert_rate of the components in one scoring run."""
        df = pd.DataFrame({"r": raw, "run": np.asarray(run_ids)})
        rank = df.groupby("run").r.rank(method="first", ascending=False)
        cap = np.floor(df.groupby("run").r.transform("size") * self.max_alert_rate)
        return ((df.r >= self.threshold) & (rank <= cap)).to_numpy()

    def level(self, raw: np.ndarray, flags: np.ndarray) -> np.ndarray:
        return np.where(flags & (raw >= self.high_threshold), "HIGH", np.where(flags, "MEDIUM", "LOW"))
