"""Step 8 - Threshold selection on validation, then a single evaluation on the test sets.

1. Build the production scorer from the Step 7 decision (single model or ensemble).
2. Calibrate probabilities on validation (isotonic) so the reported risk means "chance of an unscheduled
   removal within 30 FC".
3. Choose the alert threshold on validation with an MRO value model, subject to the hard cap of 5 alerts per
   100 active components per scoring run. Tiers: HIGH (top ~1.5%) and MEDIUM.
4. Evaluate once on test_strict (unseen aircraft and units) and test_fleet (production situation), with
   cluster-bootstrap 95% intervals, alert-rate distribution, lead time and a breakdown by segment.

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, MAX_ALERT_RATE, MIN_RECALL, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from src.metrics import EVAL_META_COLS, ci, flag_top_k, grouped_bootstrap, summarize
from src.preprocess import load_split
from src.scoring import RiskScorer

META = EVAL_META_COLS + ["tail_no", "part_no", "ata_chapter", "climate_zone", "removal_reason", "low_history",
                         "installation_id", "position"]

# Illustrative MRO value model (assumptions stated in the report)
SAVING_PER_DETECTED_USD = 30_000   # avoided AOG / delay / expedited spare when a removal is anticipated
COST_PER_INSPECTION_USD = 600      # 2-4 man-hours + access for one alert episode
HIGH_TIER_SHARE = 0.015            # HIGH = top 1.5% of components per run (valid calibration)


def build_scorer() -> RiskScorer:
    """The scorer chosen in Step 7: the ensemble decision if one was recorded, else the champion model."""
    decision_file = ARTIFACTS_DIR / "ensemble_decision.txt"
    decision = decision_file.read_text().strip() if decision_file.exists() \
        else "single: " + (MODELS_DIR / "champion.txt").read_text().strip()
    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    ranked = [m for m in comp.model if not m.startswith("Rule")]
    return RiskScorer.from_decision(decision, ranked, MODELS_DIR)


def alert_episodes(flags: np.ndarray, meta: pd.DataFrame) -> int:
    """Number of inspections: consecutive alerted runs of the same installation count as one episode."""
    d = meta[["installation_id", "snapshot_date"]].assign(f=flags).sort_values(["installation_id", "snapshot_date"])
    prev = d.groupby("installation_id").f.shift(fill_value=False)
    return int((d.f & ~prev).sum())


def value_usd(flags: np.ndarray, meta: pd.DataFrame) -> dict:
    pos = meta.assign(f=flags)[meta.label.to_numpy() == 1]
    detected = int(pos.groupby(["serial_no", "removal_date"]).f.any().sum())
    insp = alert_episodes(flags, meta)
    return {"detected_removals": detected, "inspections": insp,
            "net_value_usd": detected * SAVING_PER_DETECTED_USD - insp * COST_PER_INSPECTION_USD}


def choose_threshold(scorer: RiskScorer, raw: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """Scan thresholds on validation; the cap (5% per run) is always applied."""
    rows = []
    for q in np.linspace(0.90, 0.995, 39):
        scorer.threshold = float(np.quantile(raw, q))
        f = scorer.flag(raw, meta.snapshot_date)
        m = summarize(meta.label, raw, meta, flags=f)
        rows.append({"quantile": q, "threshold": scorer.threshold, "alert_rate": f.mean(),
                     "max_run_alert_rate": pd.Series(f).groupby(meta.snapshot_date.to_numpy()).mean().max(),
                     "event_recall": m["event_recall"], "recall": m["recall"], "precision": m["precision"],
                     **value_usd(f, meta)})
    return pd.DataFrame(rows)


def evaluate_set(name: str, scorer: RiskScorer, X: pd.DataFrame, meta: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    raw = scorer.raw_score(X)
    risk = scorer.calibrate(raw)
    flags = scorer.flag(raw, meta.snapshot_date)
    level = scorer.level(raw, flags)
    m = summarize(meta.label, raw, meta, flags=flags)
    boot = grouped_bootstrap(flags, meta, n_boot=1000)
    per_run = pd.Series(flags).groupby(meta.snapshot_date.to_numpy()).mean()
    high = level == "HIGH"
    res = {
        "set": name, "rows": len(meta), "positives": int(meta.label.sum()),
        "removal_events": int(meta[meta.label == 1].groupby(["serial_no", "removal_date"]).ngroups),
        "event_recall": m["event_recall"], "event_recall_ci": ci(boot.event_recall),
        "recall": m["recall"], "recall_ci": ci(boot.recall),
        "precision": m["precision"], "precision_ci": ci(boot.precision),
        "alert_rate_mean": per_run.mean(), "alert_rate_max_run": per_run.max(),
        "high_tier_precision": (meta.label.to_numpy()[high] == 1).mean() if high.any() else np.nan,
        "high_tier_share": high.mean(),
        "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "lead_time_fc_median": m["lead_time_fc"],
        "brier_calibrated": float(np.mean((risk - meta.label.to_numpy()) ** 2)),
        "mean_risk": float(risk.mean()), "actual_rate": float(meta.label.mean()),
        **value_usd(flags, meta),
        "meets_target": bool(m["event_recall"] >= MIN_RECALL and per_run.max() <= MAX_ALERT_RATE + 1e-9),
    }
    pred = meta.assign(raw_score=raw, risk=risk, flag=flags, risk_level=level)
    return res, pred


def segments(pred: pd.DataFrame) -> pd.DataFrame:
    pos = pred[pred.label == 1]
    ev = pos.groupby(["serial_no", "removal_date"]).agg(
        detected=("flag", "any"), ata_chapter=("ata_chapter", "first"), climate_zone=("climate_zone", "first"),
        removal_reason=("removal_reason", "first"), low_history=("low_history", "max"))
    rows = []
    for col in ["removal_reason", "ata_chapter", "climate_zone", "low_history"]:
        for k, g in ev.groupby(col, observed=True):
            rows.append({"segment": col, "value": str(k), "removal_events": len(g), "event_recall": g.detected.mean()})
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
    scorer = build_scorer()
    X_va, y_va = load_split("valid", df)
    meta_va = df.loc[X_va.index, META]

    if scorer.method == "rank":
        for n, m in scorer.models.items():
            scorer.reference[n] = np.sort(m.predict_proba(X_va)[:, 1])
    raw_va = scorer.raw_score(X_va)
    scorer.fit_calibration(raw_va, y_va.to_numpy())

    scan = choose_threshold(scorer, raw_va, meta_va)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scan.to_csv(ARTIFACTS_DIR / "threshold_scan_valid.csv", index=False)
    feasible = scan[scan.event_recall >= MIN_RECALL]
    pick = (feasible if len(feasible) else scan).sort_values("net_value_usd", ascending=False).iloc[0]
    scorer.threshold = float(pick.threshold)
    scorer.high_threshold = float(np.quantile(raw_va, 1 - HIGH_TIER_SHARE))
    scorer.metadata.update({
        "threshold_quantile_valid": float(pick["quantile"]), "valid_event_recall": float(pick.event_recall),
        "valid_alert_rate": float(pick.alert_rate), "calibrated_threshold": float(scorer.calibrate(np.array([scorer.threshold]))[0]),
        "calibrated_high_threshold": float(scorer.calibrate(np.array([scorer.high_threshold]))[0]),
        "train_period": [str(df[df.split == "train"].snapshot_date.min().date()),
                         str(df[df.split == "train"].snapshot_date.max().date())],
    })
    print(f"Scorer: {scorer.metadata['decision']}")
    print(f"Chosen on validation: threshold at the {pick['quantile']:.3f} quantile of raw scores "
          f"-> alert rate {pick.alert_rate:.2%}, event recall {pick.event_recall:.3f}, "
          f"net value ${pick.net_value_usd:,.0f}")

    results, preds = [], {}
    for name in ("test_strict", "test_fleet"):
        X, _ = load_split(name, df)
        res, pred = evaluate_set(name, scorer, X, df.loc[X.index, META])
        results.append(res)
        preds[name] = pred
    out = pd.DataFrame(results)
    out.to_csv(RESULTS_DIR / "test_results.csv", index=False)
    seg = segments(preds["test_fleet"])
    seg.to_csv(RESULTS_DIR / "test_segments.csv", index=False)

    # pure top-k curve on test for the figure
    curves = {}
    for name in ("valid", "test_strict"):
        if name == "valid":
            raw, meta = raw_va, meta_va
        else:
            raw, meta = preds[name].raw_score.to_numpy(), preds[name]
        rates = np.linspace(0.01, 0.12, 23)
        curves[name] = pd.DataFrame([{"rate": r, **{k: v for k, v in summarize(
            meta.label, raw, meta, flags=flag_top_k(raw, meta.snapshot_date, r)).items()
            if k in ("event_recall", "recall", "precision")}} for r in rates])

    joblib.dump(scorer, MODELS_DIR / "risk_scorer.joblib")
    preds["test_fleet"].to_parquet(PROCESSED_DIR / "test_predictions.parquet", index=False)
    plot_all(scan, out, curves, seg, preds["test_fleet"], scorer)

    show = out.T
    print("\nTest results (evaluated once)")
    print(show.to_string())
    print("\nEvent recall by segment (test_fleet)")
    print(seg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def plot_all(scan, out, curves, seg, pred, scorer):
    import matplotlib.pyplot as plt
    from src import viz

    pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")

    # 1. recall vs alert rate with the operating point
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c.rate, c.event_recall, color=viz.SERIES[i], label=f"event recall · {name}")
        ax.plot(c.rate, c.recall, color=viz.SERIES[i], ls="--", lw=1.5, label=f"row recall · {name}")
    r = out.set_index("set").loc["test_strict"]
    lo, hi = r.event_recall_ci
    ax.errorbar([r.alert_rate_mean], [r.event_recall], yerr=[[r.event_recall - lo], [hi - r.event_recall]],
                fmt="o", color=viz.INK, ms=7, capsize=3, zorder=5)
    ax.annotate(f"operating point (test_strict)\n{r.event_recall:.0%} event recall at {r.alert_rate_mean:.1%} alerts",
                (r.alert_rate_mean, r.event_recall), xytext=(0.065, 0.55), fontsize=9, color=viz.INK_2,
                arrowprops=dict(arrowstyle="-", color=viz.MUTED))
    ax.axhline(MIN_RECALL, color=viz.CRITICAL, ls="--", lw=1.2)
    ax.axvline(MAX_ALERT_RATE, color=viz.BASELINE, lw=1)
    ax.text(0.118, MIN_RECALL - 0.015, "80% target", ha="right", va="top", fontsize=8.5, color=viz.INK_2)
    ax.text(MAX_ALERT_RATE + 0.001, 0.03, "5 alerts / 100", fontsize=8.5, color=viz.INK_2)
    ax.set_xlabel("alert rate (share of active components flagged per scoring run)")
    ax.set_ylabel("recall")
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_ylim(0, 1)
    ax.set_title("Recall vs alert budget")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); viz.savefig(fig, "eval_recall_vs_alert_rate"); plt.close(fig)

    # 2. value curve on validation
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(scan.alert_rate, scan.net_value_usd / 1e6, color=viz.SERIES[0])
    best = scan.sort_values("net_value_usd").iloc[-1]
    chosen = scan.iloc[(scan.threshold - scorer.threshold).abs().argmin()]
    ax.plot([chosen.alert_rate], [chosen.net_value_usd / 1e6], "o", color=viz.INK, ms=7)
    ax.annotate(f"chosen: {chosen.alert_rate:.1%} alerts, \\${chosen.net_value_usd / 1e6:.2f}M",
                (chosen.alert_rate, chosen.net_value_usd / 1e6), xytext=(10, -22), textcoords="offset points",
                fontsize=9, color=viz.INK_2)
    ax.set_xlabel("mean alert rate on validation (5% cap per run always applied)")
    ax.set_ylabel("net value, \\$M (validation period)")
    ax.xaxis.set_major_formatter(pct)
    ax.set_title(f"MRO value by threshold: \\${SAVING_PER_DETECTED_USD:,} saved per anticipated removal, "
                 f"\\${COST_PER_INSPECTION_USD} per inspection", fontsize=10.5)
    fig.tight_layout(); viz.savefig(fig, "eval_value_curve"); plt.close(fig)

    # 3. calibration (test_fleet)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    bins = pd.qcut(pred.risk.rank(method="first"), 10, labels=False)
    g = pred.groupby(bins).agg(pred_mean=("risk", "mean"), actual=("label", "mean"))
    top = max(g.pred_mean.max(), g.actual.max()) * 1.1
    ax.plot([0, top], [0, top], color=viz.BASELINE, lw=1)
    ax.plot(g.pred_mean, g.actual, "o-", color=viz.SERIES[0], ms=6)
    ax.set_xlabel("predicted risk (decile mean)"); ax.set_ylabel("observed removal rate")
    ax.xaxis.set_major_formatter(pct); ax.yaxis.set_major_formatter(pct)
    ax.set_title("Calibration on test_fleet")
    fig.tight_layout(); viz.savefig(fig, "eval_calibration"); plt.close(fig)

    # 4. lead time: cycles before removal at the first alert
    pos = pred[(pred.label == 1) & pred.flag]
    first = pos.groupby(["serial_no", "removal_date"]).fc_to_removal.max()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(first, bins=np.arange(0, 32, 2), color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=2)
    ax.axvline(first.median(), color=viz.INK_2, ls="--", lw=1.2)
    ax.text(first.median() + 0.4, ax.get_ylim()[1] * 0.92, f"median {first.median():.0f} FC", fontsize=9,
            color=viz.INK_2)
    ax.set_xlabel("flight cycles before removal at the first alert"); ax.set_ylabel("detected removals")
    ax.grid(axis="x", visible=False)
    ax.set_title("Warning lead time (test_fleet)")
    fig.tight_layout(); viz.savefig(fig, "eval_lead_time"); plt.close(fig)

    # 5. segments
    s = seg[seg.segment.isin(["removal_reason", "ata_chapter"])].copy()
    s["label"] = s.apply(lambda r: f"{'reason' if r.segment == 'removal_reason' else 'ATA'} {r.value}  (n={r.removal_events})", axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    viz.barh(ax, s.label, s.event_recall, fmt="{:.0%}")
    ax.axvline(MIN_RECALL, color=viz.CRITICAL, ls="--", lw=1.2)
    ax.xaxis.set_major_formatter(pct); ax.set_xlim(0, 1.1)
    ax.set_title("Event recall by segment (test_fleet)")
    fig.tight_layout(); viz.savefig(fig, "eval_segments"); plt.close(fig)


if __name__ == "__main__":
    main()
