"""Step 5 - Leakage-safe data split.

Three layers of protection:

1. Temporal: train / valid / test are consecutive periods. Rows whose 30 FC label window would cross into the
   next period are purged, and a 7-day embargo separates periods (features overlap in time).
2. Aircraft: 8 tails (20%, stratified by aircraft type) are held out. They never appear in train or valid.
3. Component: rotables move between tails, so every train / valid row of a serial that later sits on a
   held-out tail during the test period is dropped. The model has never seen any unit it is tested on.

Test sets
  test_strict  test period x held-out tails. Unseen aircraft, unseen serials. Headline number.
  test_fleet   test period x all tails. The production situation (the fleet the model was trained on).

Usage:
    python -m src.split
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import HORIZON_FC, PROCESSED_DIR, RAW_DIR, SEED
from src.features import CycleClock

TRAIN_END = pd.Timestamp("2025-02-28")
VALID_END = pd.Timestamp("2025-06-30")
EMBARGO_DAYS = 7
N_HOLDOUT_TAILS = 8


def _quota(counts: pd.Series, n: int) -> dict:
    q = np.floor(counts / counts.sum() * n).astype(int)
    for k in (counts / counts.sum() * n - q).sort_values(ascending=False).index[: n - q.sum()]:
        q[k] += 1
    return q.to_dict()


def pick_holdout_tails(aircraft: pd.DataFrame, n: int = N_HOLDOUT_TAILS, seed: int = SEED) -> list[str]:
    """Random tails whose mix matches the fleet on both aircraft type and climate zone.

    Climate drives failure rates, so a held-out set dominated by temperate bases would flatter the model.
    """
    rng = np.random.default_rng(seed)
    type_q = _quota(aircraft.aircraft_type.value_counts(), n)
    clim_q = _quota(aircraft.climate_zone.value_counts(), n)
    for _ in range(10_000):
        pick = aircraft.iloc[rng.choice(len(aircraft), size=n, replace=False)]
        if (pick.aircraft_type.value_counts().to_dict() == type_q
                and pick.climate_zone.value_counts().to_dict() == clim_q):
            return sorted(pick.tail_no)
    raise RuntimeError("no stratified hold-out sample found")


def assign_splits(df: pd.DataFrame, usage: pd.DataFrame, holdout: list[str]) -> pd.DataFrame:
    clock = CycleClock(usage)
    d = df.snapshot_date
    valid_start = TRAIN_END + pd.Timedelta(days=EMBARGO_DAYS)
    test_start = VALID_END + pd.Timedelta(days=EMBARGO_DAYS)
    is_holdout = df.tail_no.isin(holdout)

    period = np.select([d <= TRAIN_END, (d >= valid_start) & (d <= VALID_END), d >= test_start],
                       ["train", "valid", "test"], default="embargo")
    split = pd.Series(period, index=df.index, dtype=object)

    # purge rows whose label window reaches past the end of their period
    for name, end in (("train", TRAIN_END), ("valid", VALID_END)):
        rows = split == name
        fc_left = clock.at(df.tail_no[rows], [end] * rows.sum()) - clock.at(df.tail_no[rows], d[rows])
        split[rows] = np.where(fc_left < HORIZON_FC, "purged", name)

    # held-out tails are only used in the test period
    split[is_holdout & split.isin(["train", "valid", "purged", "embargo"])] = "holdout_tail_unused"

    # component-level: drop train/valid history of serials that are tested on held-out tails
    test_serials = set(df.serial_no[(split == "test") & is_holdout])
    split[split.isin(["train", "valid"]) & df.serial_no.isin(test_serials)] = "dropped_test_serial"

    out = df.copy()
    out["split"] = split
    out["test_strict"] = (split == "test") & is_holdout
    out["test_fleet"] = split == "test"
    return out


def tail_group_folds(train: pd.DataFrame, n_splits: int = 4):
    """Cross-validation folds grouped by tail, used for hyper-parameter tuning inside the training period."""
    return list(GroupKFold(n_splits=n_splits).split(train, train.label, groups=train.tail_no))


def check_no_leakage(df: pd.DataFrame, holdout: list[str]):
    tr, va = df[df.split == "train"], df[df.split == "valid"]
    ts = df[df.test_strict]
    fit_rows = pd.concat([tr, va])
    assert not set(fit_rows.tail_no) & set(holdout), "held-out tail in train/valid"
    assert not set(fit_rows.serial_no) & set(ts.serial_no), "test serial seen in train/valid"
    assert tr.snapshot_date.max() < va.snapshot_date.min() < ts.snapshot_date.min()
    # no train/valid label window reaches the next period
    assert (tr.fc_to_removal[tr.label == 1] <= HORIZON_FC).all()
    assert (tr.snapshot_date.max() + pd.Timedelta(days=EMBARGO_DAYS)) <= va.snapshot_date.min()


def plot_timeline(df: pd.DataFrame, holdout: list[str]):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from src import viz

    cats = ["train", "valid", "test", "gap", "unused"]
    colors = [viz.SERIES[0], viz.SERIES[1], viz.SERIES[2], "#c3c2b7", "#ebeae4"]
    lab = df.split.replace({"purged": "gap", "embargo": "gap", "holdout_tail_unused": "unused",
                            "dropped_test_serial": None})
    grid = (df.assign(cat=lab).dropna(subset=["cat"])
            .groupby(["tail_no", "snapshot_date"]).cat.agg(lambda x: x.mode().iat[0])
            .map({c: i for i, c in enumerate(cats)}).unstack())
    order = [t for t in grid.index if t not in holdout] + holdout
    grid = grid.loc[order]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.imshow(grid.to_numpy(), aspect="auto", interpolation="nearest",
              cmap=ListedColormap(colors), vmin=0, vmax=len(cats) - 1)
    dates = grid.columns
    ticks = [i for i, x in enumerate(dates) if x.day <= 3 and x.month in (1, 4, 7, 10)]
    ax.set_xticks(ticks, [dates[i].strftime("%b %Y") for i in ticks])
    ax.set_yticks([len(order) - N_HOLDOUT_TAILS / 2 - 0.5, (len(order) - N_HOLDOUT_TAILS) / 2],
                  [f"{N_HOLDOUT_TAILS} held-out\ntails", f"{len(order) - N_HOLDOUT_TAILS} training\ntails"])
    ax.axhline(len(order) - N_HOLDOUT_TAILS - 0.5, color=viz.SURFACE, lw=3)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Train / validation / test split: time × aircraft")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, [
        "train", "validation", "test", "purge + embargo gap", "held-out tail, not used"])],
        loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5)
    fig.tight_layout()
    viz.savefig(fig, "split_timeline")
    plt.close(fig)


def main():
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    aircraft = pd.read_csv(RAW_DIR / "aircraft.csv")
    usage = pd.read_csv(RAW_DIR / "flight_usage.csv", parse_dates=["flight_date"])

    holdout = pick_holdout_tails(aircraft)
    df = assign_splits(df, usage, holdout)
    check_no_leakage(df, holdout)
    df.to_parquet(PROCESSED_DIR / "model_table.parquet", index=False)
    plot_timeline(df, holdout)

    print("Held-out tails:", holdout)
    print(aircraft[aircraft.tail_no.isin(holdout)].groupby(["aircraft_type", "climate_zone"]).size()
          .rename("n").to_string())
    print("\nRows by split")
    print(df.split.value_counts().to_string())
    rows = []
    for name, mask in [("train", df.split == "train"), ("valid", df.split == "valid"),
                       ("test_strict", df.test_strict), ("test_fleet", df.test_fleet)]:
        part = df[mask]
        rows.append({"set": name, "from": part.snapshot_date.min().date(), "to": part.snapshot_date.max().date(),
                     "rows": len(part), "positives": int(part.label.sum()), "pos_rate": f"{part.label.mean():.2%}",
                     "tails": part.tail_no.nunique(), "serials": part.serial_no.nunique()})
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    folds = tail_group_folds(df[df.split == "train"])
    print(f"\nGroupKFold by tail inside train: {len(folds)} folds, "
          f"validation tails per fold = {[len(set(df[df.split == 'train'].tail_no.iloc[v])) for _, v in folds]}")
    print("Leakage checks passed.")


if __name__ == "__main__":
    main()
