"""Step 6b - How does the class-imbalance strategy change the final result?

Three models (Logistic Regression, LightGBM, XGBoost). Same features, same model settings, only the imbalance handling changes. Evaluated on the validation split
(natural ~1.7% positive rate); the test sets are not touched.

Strategies
  none            train on the raw 2% data
  weight_sqrt     positive class weight = sqrt(neg/pos) ~ 7   (moderate)
  weight_full     positive class weight = neg/pos ~ 48        ("balanced")
  undersample     random negatives dropped to 1 positive : 10 negatives
  smote           synthetic positives added up to 1 : 10 (SMOTE on the imputed, scaled matrix)

Usage:
    python -m src.imbalance_experiment
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from src.config import ARTIFACTS_DIR, MAX_ALERT_RATE, PROCESSED_DIR, RESULTS_DIR, SEED
from src.metrics import EVAL_META_COLS, calibration_summary, flag_threshold, summarize
from src.models import MODEL_REGISTRY, fit_params as model_fit_params
from src.preprocess import imbalance_weights, linear_preprocessor, load_split

MODELS = ["LogisticRegression", "LightGBM", "XGBoost"]
STRATEGIES = ["none", "weight_sqrt", "weight_full", "undersample", "smote"]
SEEDS = [SEED, SEED + 1, SEED + 2]
META_COLS = EVAL_META_COLS
# Fixed, untuned settings: only the imbalance strategy varies between runs
FIXED_PARAMS = {
    "LogisticRegression": {"C": 0.5},
    "LightGBM": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50,
                 "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "XGBoost": {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 5,
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
}


def make_pipeline(model_name: str, strategy: str, pos_weight: float, seed: int):
    spec = MODEL_REGISTRY[model_name]
    w = {"none": 1.0, "weight_sqrt": np.sqrt(pos_weight), "weight_full": pos_weight}.get(strategy, 1.0)
    sampler = {"undersample": RandomUnderSampler(sampling_strategy=0.1, random_state=seed),
               "smote": SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=seed)}.get(strategy)
    # SMOTE needs a complete numeric matrix, so it always runs on the imputed (linear) branch
    prep = linear_preprocessor() if strategy == "smote" else spec.make_preprocessor()
    steps = [("prep", prep)] + ([("resample", sampler)] if sampler is not None else [])
    return ImbPipeline(steps + [("model", spec.make_estimator(FIXED_PARAMS[model_name], w, seed))])


def fit_params(model_name: str, strategy: str) -> dict:
    return {} if strategy == "smote" else model_fit_params(model_name)


def run() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
    X_tr, y_tr = load_split("train", df)
    X_va, y_va = load_split("valid", df)
    meta = df.loc[X_va.index, META_COLS]
    pos_weight = imbalance_weights(y_tr)["pos_weight"]

    rows = []
    for model_name in MODELS:
        for strategy in STRATEGIES:
            deterministic = model_name == "LogisticRegression" and strategy not in ("undersample", "smote")
            for seed in SEEDS[:1] if deterministic else SEEDS:
                t0 = time.time()
                pipe = make_pipeline(model_name, strategy, pos_weight, seed)
                pipe.fit(X_tr, y_tr, **fit_params(model_name, strategy))
                p = pipe.predict_proba(X_va)[:, 1]
                budget = summarize(y_va, p, meta)                        # top 5% per scoring run
                default = summarize(y_va, p, meta, flags=flag_threshold(p, 0.5))
                rows.append({
                    "model": model_name, "strategy": strategy, "seed": seed,
                    "pr_auc": budget["pr_auc"],
                    "recall_at_5pct": budget["recall"],
                    "precision_at_5pct": budget["precision"],
                    "event_recall_at_5pct": budget["event_recall"],
                    "recall_at_0.5": default["recall"],
                    "alert_rate_at_0.5": default["alert_rate"],
                    **calibration_summary(y_va, p),
                    "fit_seconds": time.time() - t0,
                })
                print(f"  {model_name:<18} {strategy:<12} seed {seed}  recall@5% {budget['recall']:.3f}  "
                      f"PR-AUC {budget['pr_auc']:.3f}  alert@0.5 {default['alert_rate']:.2%}")
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, actual_rate: float):
    import matplotlib.pyplot as plt
    from src import viz

    labels = {"none": "none", "weight_sqrt": "weight ×7", "weight_full": "weight ×48",
              "undersample": "undersample 1:10", "smote": "SMOTE 1:10"}
    x = np.arange(len(STRATEGIES))
    ref_label = {"alert_rate_at_0.5": "5% alert budget"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = [("recall_at_5pct", "Recall @ 5% alert budget\n(what the brief asks)", "{:.0%}", None),
              ("alert_rate_at_0.5", "Alert rate at the default 0.5 threshold", "{:.1%}", MAX_ALERT_RATE),
              ("mean_pred", f"Mean predicted probability\n(dashed = actual rate {actual_rate:.1%})", "{:.1%}", actual_rate)]
    for ax, (col, title, fmt, ref) in zip(axes, panels):
        width = 0.27
        for i, model in enumerate(MODELS):
            s = summary[summary.model == model].set_index("strategy").reindex(STRATEGIES)
            bars = ax.bar(x + (i - 1) * width, s[f"{col}_mean"], width=width * 0.94, color=viz.SERIES[i],
                          edgecolor=viz.SURFACE, linewidth=2, label=model)
            if s[f"{col}_std"].notna().any():
                ax.errorbar(x + (i - 1) * width, s[f"{col}_mean"], yerr=s[f"{col}_std"].fillna(0),
                            fmt="none", ecolor=viz.INK_2, elinewidth=1, capsize=2)
            if col == "recall_at_5pct":
                for b, v in zip(bars, s[f"{col}_mean"]):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, fmt.format(v), ha="center",
                            va="bottom", fontsize=6.5, color=viz.INK_2)
        if ref is not None:
            ax.axhline(ref, color=viz.MUTED, ls="--", lw=1.2)
            if col in ref_label:
                ax.text(len(STRATEGIES) - 0.5, ref, ref_label[col], ha="right", va="bottom",
                        fontsize=8.5, color=viz.INK_2)
        ax.set_xticks(x, [labels[s] for s in STRATEGIES], rotation=20, ha="right")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].set_ylim(0, 0.85)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Effect of class-imbalance handling (validation set)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    viz.savefig(fig, "imbalance_experiment")
    plt.close(fig)


def main():
    res = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(ARTIFACTS_DIR / "imbalance_experiment_runs.csv", index=False)
    metrics = ["pr_auc", "recall_at_5pct", "precision_at_5pct", "event_recall_at_5pct", "recall_at_0.5",
               "alert_rate_at_0.5", "mean_pred", "brier", "fit_seconds"]
    summary = res.groupby(["model", "strategy"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS_DIR / "imbalance_experiment.csv", index=False)
    plot(summary, res.actual_rate.iloc[0])

    view = summary[["model", "strategy"] + [f"{m}_mean" for m in metrics]]
    view.columns = ["model", "strategy"] + metrics
    order = {s: i for i, s in enumerate(STRATEGIES)}
    view = view.sort_values(["model", "strategy"], key=lambda c: c.map(order) if c.name == "strategy" else c)
    print("\nValidation results (mean over seeds)")
    print(view.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nActual positive rate in validation: {res.actual_rate.iloc[0]:.3%}")


if __name__ == "__main__":
    import sys
    if "--plot-only" in sys.argv:
        runs = pd.read_csv(ARTIFACTS_DIR / "imbalance_experiment_runs.csv")
        plot(pd.read_csv(RESULTS_DIR / "imbalance_experiment.csv"), runs.actual_rate.iloc[0])
    else:
        main()
