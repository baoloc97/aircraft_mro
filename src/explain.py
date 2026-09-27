"""Step 9 - Why is a component high risk?

Global: SHAP values on the test period show which features drive risk across the fleet.
Local:  for each alerted component, the top SHAP contributions are turned into plain maintenance language
        ("pressure 3.4σ below this unit's own baseline over the last 3 days", "4 related ATA 29 fault messages
        in 7 days", ...), which is what the engineer sees in the work-order recommendation and the API.

Example set: correctly caught removals (different systems), a false alert and a missed removal, so the report
shows both what the model does well and where it fails.

Usage:
    python -m src.explain
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.reasons import reason
from src.scoring import RiskScorer

TOP_N = 3


# ------------------------------------------------------------------------------------------ SHAP
def explainer_for(scorer: RiskScorer):
    """SHAP explainer for the scorer. For an ensemble, the strongest member is explained (stated in the output)."""
    name = next(iter(scorer.models)) if scorer.method == "single" else scorer.metadata.get("explain_member",
                                                                                        next(iter(scorer.models)))
    pipe = scorer.models[name]
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = list(prep.get_feature_names_out())
    if name == "LogisticRegression":
        bg = prep.transform(pd.read_parquet(PROCESSED_DIR / "model_table.parquet").query("split == 'train'")
                            .sample(2000, random_state=SEED)[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        return name, prep, names, shap.LinearExplainer(model, bg)
    return name, prep, names, shap.TreeExplainer(model)


def shap_frame(scorer: RiskScorer, X: pd.DataFrame) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    name, prep, names, ex = explainer_for(scorer)
    Z = prep.transform(X[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    sv = ex.shap_values(Z)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    return name, pd.DataFrame(sv, columns=names, index=X.index), pd.DataFrame(Z, columns=names, index=X.index)


def to_original_feature(col: str) -> str:
    """Map a transformed column (one-hot / missing flag) back to the business feature it came from."""
    if col.endswith("_missing"):
        return col.removesuffix("_missing")
    for c in CATEGORICAL_FEATURES:
        if col == c or col.startswith(c + "_"):
            return c
    return col


def collapse(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Sum SHAP over the columns that belong to one business feature (e.g. all one-hot levels of part_no)."""
    groups = {c: to_original_feature(c) for c in shap_df.columns}
    return shap_df.T.groupby(groups).sum().T


def top_factors(shap_row: pd.Series, row: pd.Series, n: int = TOP_N) -> list[dict]:
    top = shap_row.sort_values(ascending=False)
    top = top[top > 0].head(n)
    out = []
    for f, s in top.items():
        val = row.get(f, np.nan)
        out.append({"feature": f, "value": None if pd.isna(val) else (float(val) if np.isscalar(val) and not isinstance(val, str) else str(val)),
                    "impact": round(float(s), 3), "reason": reason(f, row)})
    return out


# ------------------------------------------------------------------------------------------ main
def main():
    scorer: RiskScorer = joblib.load(MODELS_DIR / "risk_scorer.joblib")
    pred = pd.read_parquet(PROCESSED_DIR / "test_predictions.parquet")
    df = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
    feats = df.loc[df.test_fleet].reset_index(drop=True)
    assert len(feats) == len(pred)
    feats = pd.concat([feats, pred[["raw_score", "risk", "flag", "risk_level"]]], axis=1)

    # global SHAP on a sample: all positives + flagged rows + random negatives
    rng = np.random.default_rng(SEED)
    idx = np.unique(np.concatenate([np.where(feats.label == 1)[0], np.where(feats.flag)[0],
                                    rng.choice(len(feats), 6000, replace=False)]))
    sample = feats.iloc[idx]
    member, sv, Z = shap_frame(scorer, sample)
    svc = collapse(sv)
    imp = svc.abs().mean().sort_values(ascending=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    imp.rename("mean_abs_shap").to_csv(RESULTS_DIR / "shap_importance.csv")
    plot_global(imp, svc, sample)

    # example components
    flagged_tp = feats[(feats.risk_level == "HIGH") & (feats.label == 1)]
    examples = []
    for ata in flagged_tp.ata_chapter.astype(str).value_counts().index[:3]:
        examples.append(("caught removal", flagged_tp[flagged_tp.ata_chapter.astype(str) == ata]
                         .sort_values("risk", ascending=False).index[0]))
    fp = feats[(feats.risk_level == "HIGH") & (feats.label == 0) & feats.removal_type.isna()]
    if len(fp):
        examples.append(("false alert", fp.sort_values("risk", ascending=False).index[0]))
    ev = feats[feats.label == 1].groupby(["serial_no", "removal_date"]).flag.transform("any")
    missed = feats[(feats.label == 1) & ~ev.reindex(feats.index, fill_value=True)]
    if len(missed):
        examples.append(("missed removal", missed.sort_values("fc_to_removal").index[0]))

    ex_rows = feats.loc[[i for _, i in examples]]
    _, ex_sv, _ = shap_frame(scorer, ex_rows)
    ex_sv = collapse(ex_sv)
    cards = []
    for (kind, i) in examples:
        row = feats.loc[i]
        cards.append({
            "case": kind,
            "snapshot_date": str(row.snapshot_date.date()),
            "serial_no": row.serial_no, "part_no": row.part_no, "ata_chapter": str(row.ata_chapter),
            "tail_no": row.tail_no, "position": row.position,
            "risk_score": round(float(row.risk), 3), "risk_level": row.risk_level,
            "actual_outcome": (f"unscheduled removal ({row.removal_reason}) {int(row.fc_to_removal)} FC later"
                               if row.label == 1 else "no unscheduled removal within 30 FC"),
            "top_factors": top_factors(ex_sv.loc[i], row),
            "risk_reducing_factors": [{"feature": f, "impact": round(float(s), 3), "reason": reason(f, row)}
                                      for f, s in ex_sv.loc[i].sort_values().head(2).items() if s < 0],
        })
    (RESULTS_DIR / "example_explanations.json").write_text(json.dumps(
        {"explained_model": member, "examples": cards}, indent=2, default=str))
    plot_waterfalls(ex_sv, examples, feats)

    print(f"Explained model: {member}")
    print("\nGlobal importance (mean |SHAP|, top 12):")
    print(imp.head(12).round(3).to_string())
    for c in cards:
        print(f"\n[{c['case']}] {c['serial_no']} {c['part_no']} (ATA {c['ata_chapter']}) on {c['tail_no']} "
              f"{c['position']} · {c['snapshot_date']} · risk {c['risk_score']} {c['risk_level']}")
        print(f"   outcome: {c['actual_outcome']}")
        for f in c["top_factors"]:
            print(f"   + {f['reason']}  (SHAP {f['impact']:+.2f})")
        for f in c["risk_reducing_factors"]:
            print(f"   - {f['reason']}  (SHAP {f['impact']:+.2f})")


def plot_global(imp: pd.Series, svc: pd.DataFrame, sample: pd.DataFrame):
    import matplotlib.pyplot as plt
    from src import viz

    top = imp.head(15)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
    viz.barh(axes[0], top.index, top.values, fmt="{:.2f}")
    axes[0].set_title("Global importance: mean |SHAP| (log-odds)")

    ax = axes[1]
    feats = top.index[:10][::-1]
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("coolwarm")
    for i, f in enumerate(feats):
        s = svc[f].to_numpy()
        v = pd.to_numeric(sample[f], errors="coerce") if f in sample else pd.Series(np.nan, index=sample.index)
        r = v.rank(pct=True).to_numpy()
        keep = rng.choice(len(s), min(2500, len(s)), replace=False)
        ax.scatter(s[keep], i + rng.normal(0, 0.12, len(keep)), c=np.nan_to_num(r[keep], nan=0.5), cmap=cmap,
                   s=5, alpha=0.6, linewidths=0)
    ax.axvline(0, color=viz.BASELINE, lw=1)
    ax.set_yticks(range(len(feats)), feats)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("SHAP value (impact on log-odds of removal)")
    ax.set_title("Direction: red = high feature value, blue = low")
    fig.tight_layout(); viz.savefig(fig, "shap_global"); plt.close(fig)


def plot_waterfalls(ex_sv: pd.DataFrame, examples, feats):
    import matplotlib.pyplot as plt
    from src import viz

    ncol = 3
    nrow = int(np.ceil(len(examples) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.9 * nrow), squeeze=False)
    for ax in axes.flat[len(examples):]:
        ax.set_visible(False)
    for ax, (kind, i) in zip(axes.flat, examples):
        s = ex_sv.loc[i]
        top = s.reindex(s.abs().sort_values(ascending=False).index[:7])[::-1]
        ax.barh(top.index, top.values, color=[viz.CRITICAL if v > 0 else viz.SERIES[0] for v in top.values],
                height=0.6, edgecolor=viz.SURFACE, linewidth=1.5)
        ax.axvline(0, color=viz.BASELINE, lw=1)
        ax.grid(axis="y", visible=False)
        row = feats.loc[i]
        ax.set_title(f"{kind}\n{row.part_no} · risk {row.risk:.2f} ({row.risk_level})", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Per-component explanations (red raises risk, blue lowers it)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); viz.savefig(fig, "shap_examples"); plt.close(fig)


if __name__ == "__main__":
    main()
