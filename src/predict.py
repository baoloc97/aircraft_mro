"""Step 10 - Score the fleet as of any date, the way the daily production job would.

Only data dated <= the scoring date is used (raw tables are truncated first), every active installation is
scored, and each result carries the risk score, the risk level, the recommended action, the top reasons and
data-quality flags (new component, missing sensor data).

Usage:
    python -m src.predict --date 2025-10-15              # writes artifacts/fleet_risk_2025-10-15.csv
    python -m src.predict --date 2025-10-15 --top 10
"""
from __future__ import annotations

import argparse
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, MODELS_DIR
from src.explain import collapse, shap_frame, top_factors
from src.features import build_features, load_raw, raw_as_of
from src.scoring import RiskScorer

ACTIONS = {
    "HIGH": "Inspect within the next 5 flight cycles; pre-position a serviceable spare",
    "MEDIUM": "Inspect at the next daily/weekly check; review trend and fault history",
    "LOW": "No action; keep monitoring",
}


@lru_cache(maxsize=1)
def load_scorer() -> RiskScorer:
    return joblib.load(MODELS_DIR / "risk_scorer.joblib")


@lru_cache(maxsize=1)
def _raw():
    return load_raw()


def active_snapshots(raw: dict, t: pd.Timestamp) -> pd.DataFrame:
    ins = raw["ins"]
    active = ins[(ins.install_date <= t) & ins.removal_date.isna()]
    snaps = active[["installation_id", "serial_no", "part_no", "tail_no", "position", "install_date"]].copy()
    snaps.insert(0, "snapshot_date", t)
    return snaps.reset_index(drop=True)


def score_frame(feats: pd.DataFrame, explain: bool = True) -> pd.DataFrame:
    """Score feature rows (one scoring run) and attach levels, actions, reasons and data-quality flags."""
    scorer = load_scorer()
    raw_score = scorer.raw_score(feats)
    out = feats.copy()
    out["risk_score"] = np.round(scorer.calibrate(raw_score), 4)
    out["raw_score"] = raw_score
    out["flag"] = scorer.flag(raw_score, feats.snapshot_date)
    out["risk_level"] = scorer.level(raw_score, out.flag.to_numpy())
    out["recommended_action"] = out.risk_level.map(ACTIONS)
    out["low_history"] = feats.low_history.astype(bool)
    out["sensor_missing_ratio_7d"] = feats.sensor_missing_ratio_7d
    out["days_since_last_reading"] = feats.days_since_last_reading
    if explain:
        _, sv, _ = shap_frame(scorer, feats)
        sv = collapse(sv)
        out["top_factors"] = [top_factors(sv.loc[i], feats.loc[i]) for i in feats.index]
    return out


def score_fleet(date: str, explain: bool = True) -> pd.DataFrame:
    t = pd.Timestamp(date)
    raw = raw_as_of(_raw(), t)
    snaps = active_snapshots(raw, t)
    feats = build_features(snaps, raw)
    return score_frame(feats, explain=explain).sort_values("raw_score", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    res = score_fleet(args.date)
    cols = ["snapshot_date", "tail_no", "position", "serial_no", "part_no", "ata_chapter", "risk_score",
            "risk_level", "recommended_action", "low_history", "sensor_missing_ratio_7d"]
    export = res[cols].copy()
    export["top_reasons"] = res.top_factors.map(lambda fs: " | ".join(f["reason"] for f in fs))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"fleet_risk_{args.date}.csv"
    export.to_csv(path, index=False)
    n = len(res)
    print(f"Scored {n} active components as of {args.date}: "
          f"{(res.risk_level == 'HIGH').sum()} HIGH, {(res.risk_level == 'MEDIUM').sum()} MEDIUM "
          f"({res.flag.mean():.1%} alerted)  -> {path}")
    for _, r in res.head(args.top).iterrows():
        print(f"  {r.risk_level:<6} {r.risk_score:.3f}  {r.tail_no} {r.position:<14} {r.serial_no}  {r.part_no}")
        for f in r.top_factors:
            print(f"         - {f['reason']}")


if __name__ == "__main__":
    main()
