"""Step 7 - Train, tune and compare models.

  * 2 rule baselines that mirror current MRO practice (no ML)
  * 5 ML models: Logistic Regression, Random Forest, HistGradientBoosting, LightGBM, XGBoost

Tuning: random search on the training split with GroupKFold by tail (4 folds), selecting on the metric the
brief asks for: recall at a 5% alert budget per scoring run. Positive class weight is searched in
{1, sqrt(neg/pos)} only (see the imbalance ablation).

Comparison: every tuned model is refitted on the full training split and scored on the validation split,
with 95% cluster-bootstrap intervals (resampling serial numbers). The test sets are not used here.

Usage:
    python -m src.train            # full run (~10-15 min)
    python -m src.train --fast     # fewer search iterations, for a quick check
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from src.config import ARTIFACTS_DIR, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from src.metrics import EVAL_META_COLS, ci, flag_top_k, grouped_bootstrap, summarize
from src.models import MODEL_REGISTRY, RuleBaseline, build_model, fit_params
from src.preprocess import imbalance_weights, load_split
from src.split import tail_group_folds

META_COLS = EVAL_META_COLS + ["tail_no"]


def cv_score(name: str, params: dict, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
             folds, pos_weight: float) -> tuple[float, float]:
    rec, ap = [], []
    for tr_idx, va_idx in folds:
        m = build_model(name, params, pos_weight)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], **fit_params(name))
        p = m.predict_proba(X.iloc[va_idx])[:, 1]
        s = summarize(y.iloc[va_idx], p, meta.iloc[va_idx])
        rec.append(s["recall"]); ap.append(s["pr_auc"])
    return float(np.mean(rec)), float(np.mean(ap))


def tune(name: str, X, y, meta, folds, pos_weight: float, n_iter: int) -> tuple[dict, pd.DataFrame]:
    rows = []
    for i, params in enumerate(ParameterSampler(MODEL_REGISTRY[name].search_space, n_iter=n_iter, random_state=SEED)):
        t0 = time.time()
        rec, ap = cv_score(name, params, X, y, meta, folds, pos_weight)
        rows.append({"model": name, "params": json.dumps(params), "cv_recall_at_5pct": rec, "cv_pr_auc": ap,
                     "seconds": time.time() - t0})
        print(f"    [{i + 1:>2}/{n_iter}] cv recall@5% {rec:.3f}  PR-AUC {ap:.3f}  {params}")
    res = pd.DataFrame(rows).sort_values(["cv_recall_at_5pct", "cv_pr_auc"], ascending=False)
    return json.loads(res.iloc[0].params), res


# ------------------------------------------------------------------------------------------ main
def main(fast: bool = False):
    df = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
    X_tr, y_tr = load_split("train", df)
    X_va, y_va = load_split("valid", df)
    meta_tr, meta_va = df.loc[X_tr.index, META_COLS], df.loc[X_va.index, META_COLS]
    w = imbalance_weights(y_tr)["pos_weight_sqrt"]
    folds = tail_group_folds(df.loc[X_tr.index])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    models, search_logs, val_scores = {}, [], {}
    for kind, label in (("age", "Rule: age (CSI/MTBUR)"), ("faults", "Rule: repeat faults")):
        models[label] = RuleBaseline(kind)
        val_scores[label] = models[label].predict_proba(X_va)[:, 1]

    for name, spec in MODEL_REGISTRY.items():
        print(f"\n== {name}")
        n_iter = 3 if fast else spec.n_iter
        best, log = tune(name, X_tr, y_tr, meta_tr, folds, w, n_iter)
        search_logs.append(log)
        t0 = time.time()
        model = build_model(name, best, w).fit(X_tr, y_tr, **fit_params(name))
        fit_s = time.time() - t0
        models[name] = model
        val_scores[name] = model.predict_proba(X_va)[:, 1]
        joblib.dump({"model": model, "params": best, "fit_seconds": fit_s}, MODELS_DIR / f"{name}.joblib")
        print(f"   best {best}  (refit {fit_s:.1f}s)")

    # validation comparison with paired cluster bootstrap
    flags = {k: flag_top_k(s, meta_va.snapshot_date) for k, s in val_scores.items()}
    boot = grouped_bootstrap(flags[next(iter(flags))], meta_va, n_boot=1000,
                             others={k: v for k, v in flags.items()})
    boot = boot[boot.model != "model"]
    rows = []
    for name, s in val_scores.items():
        m = summarize(y_va, s, meta_va, flags=flags[name])
        b = boot[boot.model == name]
        log = next((l for l in search_logs if l.model.iloc[0] == name), None)
        rows.append({
            "model": name,
            "cv_recall_at_5pct": None if log is None else log.cv_recall_at_5pct.iloc[0],
            "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
            "recall_at_5pct": m["recall"], "recall_ci": ci(b.recall),
            "precision_at_5pct": m["precision"],
            "event_recall": m["event_recall"], "event_recall_ci": ci(b.event_recall),
            "lead_time_fc": m["lead_time_fc"],
        })
    comp = pd.DataFrame(rows).sort_values("recall_at_5pct", ascending=False)
    top_valid = comp[~comp.model.str.startswith("Rule")].iloc[0].model

    # paired difference: top validation model vs every other
    piv = boot.pivot(index="boot", columns="model", values="recall")
    comp["delta_vs_best"] = [piv[top_valid].sub(piv[n]).mean() for n in comp.model]
    comp["p_best_better"] = [(piv[top_valid] > piv[n]).mean() if n != top_valid else np.nan for n in comp.model]

    # Champion rule: among ML models statistically tied with the top validation model (it beats them in
    # < 90% of paired resamples), take the best grouped-CV recall. CV is independent of the validation set,
    # which is also used later for calibration and the threshold.
    tied = comp[~comp.model.str.startswith("Rule") & ((comp.p_best_better < 0.90) | comp.p_best_better.isna())]
    best_name = tied.sort_values(["cv_recall_at_5pct", "pr_auc"], ascending=False).iloc[0].model
    comp["champion"] = comp.model == best_name

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    comp.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    pd.concat(search_logs).to_csv(ARTIFACTS_DIR / "tuning_log.csv", index=False)
    np.savez_compressed(ARTIFACTS_DIR / "valid_scores.npz", index=X_va.index.to_numpy(),
                        **{k.replace(" ", "_").replace(":", "").replace("/", "_"): v for k, v in val_scores.items()})
    plot_comparison(comp, val_scores, meta_va)

    show = comp.copy()
    for c in ("recall_ci", "event_recall_ci"):
        show[c] = show[c].map(lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]")
    print("\nValidation comparison (alerts = top 5% per scoring run)")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nBest ML model on validation: {best_name}")
    (MODELS_DIR / "champion.txt").write_text(best_name)


def plot_comparison(comp: pd.DataFrame, val_scores: dict, meta: pd.DataFrame):
    import matplotlib.pyplot as plt
    from src import viz

    order = comp.model.tolist()[::-1]
    c = comp.set_index("model").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})

    ax = axes[0]
    y = np.arange(len(order))
    colors = [viz.MUTED if n.startswith("Rule") else viz.SERIES[0] for n in order]
    for col, off, alpha, lab in (("recall_at_5pct", 0.18, 1.0, "row recall"),
                                 ("event_recall", -0.18, 0.55, "event recall")):
        ci_col = "recall_ci" if col == "recall_at_5pct" else "event_recall_ci"
        lo = c[col] - c[ci_col].map(lambda t: t[0]); hi = c[ci_col].map(lambda t: t[1]) - c[col]
        ax.barh(y + off, c[col], height=0.34, color=colors, alpha=alpha, edgecolor=viz.SURFACE, linewidth=1.5,
                label=lab)
        ax.errorbar(c[col], y + off, xerr=[lo, hi], fmt="none", ecolor=viz.INK_2, elinewidth=1, capsize=2)
        for yy, v, top in zip(y + off, c[col], c[ci_col].map(lambda t: t[1])):
            ax.text(top + 0.015, yy, f"{v:.0%}", va="center", fontsize=8, color=viz.INK_2)
    ax.axvline(0.8, color=viz.CRITICAL, ls="--", lw=1.2)
    ax.text(0.8, len(order) - 0.4, " 80% target", color=viz.INK_2, fontsize=8.5, va="bottom")
    champ = comp.loc[comp.get("champion", pd.Series(False, index=comp.index)).astype(bool), "model"]
    ax.set_yticks(y, [f"{n}  ★" if n in set(champ) else n for n in order])
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Recall at 5% alert budget (validation, 95% CI)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=viz.SERIES[0], label="row recall"),
                       Patch(color=viz.SERIES[0], alpha=0.55, label="event recall"),
                       Patch(color=viz.MUTED, label="rule baselines")], loc="lower right", fontsize=8.5)
    if len(champ):
        ax.set_title("Recall at 5% alert budget (validation, 95% CI, ★ = champion)")

    ax = axes[1]
    rates = np.linspace(0.005, 0.15, 30)
    ml = [m for m in comp.model if not m.startswith("Rule")]
    for i, name in enumerate(ml + [m for m in comp.model if m.startswith("Rule")]):
        s = val_scores[name]
        rec = [summarize(meta.label, s, meta, flags=flag_top_k(s, meta.snapshot_date, r))["event_recall"]
               for r in rates]
        style = dict(color=viz.SERIES[i], lw=2) if name in ml else dict(color=viz.MUTED, lw=1.5,
                                                                       ls=":" if "age" in name else "--")
        ax.plot(rates, rec, label=name, **style)
    ax.axvline(0.05, color=viz.BASELINE, lw=1)
    ax.axhline(0.8, color=viz.CRITICAL, ls="--", lw=1.2)
    ax.text(0.052, 0.03, "5% budget", fontsize=8.5, color=viz.INK_2)
    ax.set_xlabel("alert rate (share of active components flagged per scoring run)")
    ax.set_ylabel("event recall")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1)
    ax.set_title("Event recall vs alert rate (validation)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    viz.savefig(fig, "model_comparison")
    plt.close(fig)


def replot():
    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    for c in ("recall_ci", "event_recall_ci"):
        comp[c] = comp[c].map(lambda t: tuple(float(x) for x in t.strip("()").split(",")))
    npz = np.load(ARTIFACTS_DIR / "valid_scores.npz")
    key = lambda k: k.replace(" ", "_").replace(":", "").replace("/", "_")
    scores = {m: npz[key(m)] for m in comp.model}
    df = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
    plot_comparison(comp, scores, df.loc[npz["index"], META_COLS].reset_index(drop=True))


if __name__ == "__main__":
    replot() if "--plot-only" in sys.argv else main(fast="--fast" in sys.argv)
