"""Step 7b - Does voting / ensembling beat the best single model?

Candidates (validation set, alerts = top 5% per scoring run):
  single       the best single model from Step 7
  hard vote    each model flags its own top 5%; score = number of votes (0..5)
  soft vote    mean of predicted probabilities
  rank vote    mean of per-model percentile ranks (robust to different probability scales)

Decision rule, fixed before looking at the numbers: adopt an ensemble only if it beats the single model in
at least 90% of paired cluster-bootstrap resamples on event recall. Otherwise keep one model: it is simpler
to explain (SHAP), to monitor and to run.

Usage:
    python -m src.ensemble
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, MAX_ALERT_RATE, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from src.metrics import EVAL_META_COLS, ci, flag_top_k, grouped_bootstrap, summarize

ML_MODELS = ["LogisticRegression", "RandomForest", "HistGradientBoosting", "LightGBM", "XGBoost"]
META = EVAL_META_COLS
ADOPT_IF_P_BETTER = 0.90


def main():
    df = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
    npz = np.load(ARTIFACTS_DIR / "valid_scores.npz")
    meta = df.loc[npz["index"], META].reset_index(drop=True)
    scores = {m: npz[m] for m in ML_MODELS}
    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    ranked = [m for m in comp.model if m in ML_MODELS]
    champion = (MODELS_DIR / "champion.txt").read_text().strip()
    dates = meta.snapshot_date

    pct = {m: pd.Series(s).rank(pct=True).to_numpy() for m, s in scores.items()}
    votes = sum(flag_top_k(scores[m], dates).astype(int) for m in ML_MODELS)
    cands = {
        f"single: {champion}": scores[champion],
        "hard vote (5 models)": votes + 1e-6 * pct[champion],   # champion only breaks exact ties
        "soft vote: top 3": np.mean([scores[m] for m in ranked[:3]], axis=0),
        "rank vote: top 3": np.mean([pct[m] for m in ranked[:3]], axis=0),
        "rank vote: all 5": np.mean([pct[m] for m in ML_MODELS], axis=0),
        "rank vote: LightGBM + XGBoost + LR": np.mean([pct[m] for m in ["LightGBM", "XGBoost",
                                                                        "LogisticRegression"]], axis=0),
    }
    flags = {k: flag_top_k(s, dates) for k, s in cands.items()}
    boot = grouped_bootstrap(flags[f"single: {champion}"], meta, n_boot=1000, others=flags)
    boot = boot[boot.model != "model"]
    base = f"single: {champion}"
    piv_r = boot.pivot(index="boot", columns="model", values="recall")
    piv_e = boot.pivot(index="boot", columns="model", values="event_recall")

    # how many components tie at the 5% cut-off under hard voting
    cut = []
    for d, g in pd.DataFrame({"v": votes, "d": dates}).groupby("d"):
        k = int(np.floor(len(g) * MAX_ALERT_RATE))
        kth = np.sort(g.v.to_numpy())[::-1][k - 1]
        cut.append(((g.v == kth).sum(), (g.v > kth).sum(), k))
    cut = pd.DataFrame(cut, columns=["tied_at_cutoff", "above_cutoff", "k"])

    rows = []
    for name, s in cands.items():
        m = summarize(meta.label, s, meta, flags=flags[name])
        rows.append({"candidate": name, "pr_auc": m["pr_auc"], "recall_at_5pct": m["recall"],
                     "event_recall": m["event_recall"], "event_recall_ci": ci(piv_e[name]),
                     "delta_event_recall": (piv_e[name] - piv_e[base]).mean(),
                     "p_better_event": (piv_e[name] > piv_e[base]).mean() if name != base else np.nan,
                     "delta_recall": (piv_r[name] - piv_r[base]).mean(),
                     "p_better_recall": (piv_r[name] > piv_r[base]).mean() if name != base else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "ensemble_experiment.csv", index=False)

    corr = pd.DataFrame(pct).corr(method="spearman")
    print("Spearman correlation of model scores (validation):")
    print(corr.round(3).to_string())
    print(f"\nHard vote: on a typical scoring run {cut.tied_at_cutoff.median():.0f} components tie at the cut-off "
          f"for {cut.k.median():.0f} alert slots ({cut.above_cutoff.median():.0f} clearly above it).")
    show = out.copy()
    show["event_recall_ci"] = show.event_recall_ci.map(lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]")
    print("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    winners = out[(out.p_better_event >= ADOPT_IF_P_BETTER)]
    decision = winners.sort_values("delta_event_recall", ascending=False).candidate.iloc[0] if len(winners) else base
    print(f"\nDecision (adopt only if P(better) >= {ADOPT_IF_P_BETTER:.0%} on event recall): {decision}")
    (ARTIFACTS_DIR / "ensemble_decision.txt").write_text(decision)


if __name__ == "__main__":
    main()
