"""Evaluation metrics tied to the MRO constraint: >= 80% of removals caught with <= 5 alerts per 100 components.

Alert budget is applied per scoring run (snapshot date): at most 5% of the components active on that date
can be flagged. This is the operational meaning of "5 alerts per 100 active components" and does not depend
on how often the model is run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.config import MAX_ALERT_RATE

# Columns every evaluation needs next to the scores (dates for the per-run budget, keys for event recall)
EVAL_META_COLS = ["snapshot_date", "serial_no", "removal_date", "fc_to_removal", "label"]


def flag_top_k(scores: np.ndarray, dates: pd.Series, rate: float = MAX_ALERT_RATE) -> np.ndarray:
    """Flag the highest-scoring `rate` share of components on each snapshot date (capacity-constrained)."""
    df = pd.DataFrame({"s": scores, "d": np.asarray(dates)})
    rank = df.groupby("d").s.rank(method="first", ascending=False)
    k = np.floor(df.groupby("d").s.transform("size") * rate)
    return (rank <= k).to_numpy()


def flag_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores) >= threshold


def event_recall(flags: np.ndarray, meta: pd.DataFrame) -> float:
    """Share of unscheduled removals flagged at least once in their 30 FC window."""
    pos = meta.assign(flag=flags)[meta.label.to_numpy() == 1]
    return pos.groupby(["serial_no", "removal_date"]).flag.any().mean()


def lead_time_fc(flags: np.ndarray, meta: pd.DataFrame) -> float:
    """Median flight cycles between the first alert and the removal, over detected removals."""
    pos = meta.assign(flag=flags)[(meta.label.to_numpy() == 1) & flags]
    if pos.empty:
        return float("nan")
    return pos.groupby(["serial_no", "removal_date"]).fc_to_removal.max().median()


def summarize(y: pd.Series, scores: np.ndarray, meta: pd.DataFrame, flags: np.ndarray | None = None) -> dict:
    """Headline metrics. `meta` needs snapshot_date, serial_no, removal_date, fc_to_removal, label."""
    y = np.asarray(y)
    if flags is None:
        flags = flag_top_k(scores, meta.snapshot_date)
    tp = (flags & (y == 1)).sum()
    return {
        "pr_auc": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "recall": tp / max(y.sum(), 1),
        "precision": tp / max(flags.sum(), 1),
        "alert_rate": flags.mean(),
        "event_recall": event_recall(flags, meta),
        "lead_time_fc": lead_time_fc(flags, meta),
    }


def calibration_summary(y: pd.Series, proba: np.ndarray) -> dict:
    return {"mean_pred": float(np.mean(proba)), "actual_rate": float(np.mean(y)),
            "brier": brier_score_loss(y, proba)}


def grouped_bootstrap(flags: np.ndarray, meta: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
                      others: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Cluster bootstrap of recall, precision and event recall with the alerts held fixed.

    Rows are not independent (consecutive snapshots of one unit), so we resample whole serial numbers.
    Alerts are computed once on the full set, exactly as they would be in operation; the bootstrap only
    measures how much the metrics depend on which units happened to be observed.

    `others` maps a model name to its own flags; the same resamples are used for every model, so
    differences between models can be read as paired differences.
    """
    rng = np.random.default_rng(seed)
    y = meta.label.to_numpy()
    serial_codes, serials = pd.factorize(meta.serial_no)
    event_key = pd.factorize(meta.serial_no.astype(str) + "|" + meta.removal_date.astype(str))[0]
    all_flags = {"model": flags, **(others or {})}

    # per-serial sufficient statistics, so each resample is a weighted sum (fast)
    stats = {}
    for name, f in all_flags.items():
        tp = np.bincount(serial_codes, weights=(f & (y == 1)), minlength=len(serials))
        fl = np.bincount(serial_codes, weights=f, minlength=len(serials))
        ev = pd.DataFrame({"serial": serial_codes, "event": event_key, "f": f})[y == 1]
        ev = ev.groupby(["serial", "event"]).f.any().groupby(level="serial").agg(["sum", "size"])
        det = np.zeros(len(serials)); tot = np.zeros(len(serials))
        det[ev.index] = ev["sum"]; tot[ev.index] = ev["size"]
        stats[name] = (tp, fl, det, tot)
    pos = np.bincount(serial_codes, weights=(y == 1), minlength=len(serials))

    rows = []
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(serials), len(serials)), minlength=len(serials))
        for name, (tp, fl, det, tot) in stats.items():
            rows.append({"boot": b, "model": name,
                         "recall": (w @ tp) / max(w @ pos, 1),
                         "precision": (w @ tp) / max(w @ fl, 1),
                         "event_recall": (w @ det) / max(w @ tot, 1)})
    return pd.DataFrame(rows)


def ci(values: pd.Series, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(values.quantile(a)), float(values.quantile(1 - a))
