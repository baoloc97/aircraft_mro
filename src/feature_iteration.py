"""Step 7a - Does the second feature iteration help? Grouped CV on train + validation, same LightGBM settings.

Usage:
    python -m src.feature_iteration
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

from src.config import PROCESSED_DIR, RESULTS_DIR, SEED
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, V2_FEATURES
from src.metrics import EVAL_META_COLS, summarize
from src.split import tail_group_folds

META = EVAL_META_COLS
PARAMS = dict(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=50, subsample=0.85,
              subsample_freq=1, reg_lambda=5.0, colsample_bytree=1.0, random_state=SEED, verbose=-1, n_jobs=-1)


def pipe(num):
    ct = ColumnTransformer([("num", "passthrough", num),
                            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                                   encoded_missing_value=np.nan), CATEGORICAL_FEATURES)])
    return Pipeline([("prep", ct), ("model", lgb.LGBMClassifier(**PARAMS))]), \
        {"model__categorical_feature": list(range(len(num), len(num) + len(CATEGORICAL_FEATURES)))}


def main():
    df = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
    tr, va = df[df.split == "train"], df[df.split == "valid"]
    folds = tail_group_folds(tr)
    sets = {"v1 (42 features)": [c for c in NUMERIC_FEATURES if c not in V2_FEATURES],
            "v2 (+ short-window z, 30-day slope, acceleration)": NUMERIC_FEATURES}
    rows = []
    for name, num in sets.items():
        cv = []
        for a, b in folds:
            m, fp = pipe(num)
            m.fit(tr.iloc[a][num + CATEGORICAL_FEATURES], tr.label.iloc[a], **fp)
            p = m.predict_proba(tr.iloc[b][num + CATEGORICAL_FEATURES])[:, 1]
            cv.append(summarize(tr.label.iloc[b], p, tr.iloc[b][META]))
        m, fp = pipe(num)
        m.fit(tr[num + CATEGORICAL_FEATURES], tr.label, **fp)
        v = summarize(va.label, m.predict_proba(va[num + CATEGORICAL_FEATURES])[:, 1], va[META])
        rows.append({"feature_set": name,
                     "cv_recall_at_5pct": np.mean([c["recall"] for c in cv]),
                     "cv_recall_fold_min": np.min([c["recall"] for c in cv]),
                     "cv_event_recall": np.mean([c["event_recall"] for c in cv]),
                     "cv_pr_auc": np.mean([c["pr_auc"] for c in cv]),
                     "valid_recall_at_5pct": v["recall"], "valid_event_recall": v["event_recall"],
                     "valid_pr_auc": v["pr_auc"]})
    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "feature_iteration.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
