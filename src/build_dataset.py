"""Step 3 - Build the snapshot table and the label.

One row = one installed component (serial number in a tail/position) at the end of a snapshot day t.

    label = 1  if the installation ends with an UNSCHEDULED removal (confirmed failure or NFF)
               after day t, and the aircraft flies <= HORIZON_FC cycles between t and the removal day
    label = 0  otherwise

Rules
  * a unit is "active" on day t if install_date <= t < removal_date: a unit removed on day t is not scored
  * scheduled removals (wear limit, opportunistic) are planned work, so they are negatives
  * right censoring: snapshots whose aircraft flies < HORIZON_FC cycles before the end of the data are dropped,
    because their label is not yet known. Positives are dropped too, so the end of the period is not biased
  * removal_type / removal_reason / fc_to_removal are kept only as metadata for evaluation, never as features

Usage:
    python -m src.build_dataset
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HORIZON_FC, PROCESSED_DIR, RAW_DIR

SNAPSHOT_EVERY_DAYS = 3
WARMUP_DAYS = 28          # first snapshot after four weeks, so rolling features have history

KEY_COLS = ["snapshot_date", "installation_id", "serial_no", "part_no", "tail_no", "position"]
META_COLS = ["install_date", "removal_date", "removal_type", "removal_reason", "fc_to_removal",
             "days_to_removal", "followup_fc"]


def load_raw():
    ins = pd.read_csv(RAW_DIR / "installations.csv", parse_dates=["install_date", "removal_date"])
    rem = pd.read_csv(RAW_DIR / "removals.csv", parse_dates=["removal_date"])
    usage = pd.read_csv(RAW_DIR / "flight_usage.csv", parse_dates=["flight_date"])
    return ins, rem, usage


def cumulative_cycles(usage: pd.DataFrame) -> pd.DataFrame:
    """Tail x date matrix of cumulative flight cycles flown up to the end of each day."""
    return (usage.pivot(index="flight_date", columns="tail_no", values="flight_cycles")
            .sort_index().cumsum())


def build_snapshots(ins: pd.DataFrame, rem: pd.DataFrame, usage: pd.DataFrame,
                    every_days: int = SNAPSHOT_EVERY_DAYS) -> pd.DataFrame:
    cum = cumulative_cycles(usage)
    first, last = cum.index.min(), cum.index.max()
    dates = pd.date_range(first + pd.Timedelta(days=WARMUP_DAYS), last, freq=f"{every_days}D")

    # expand each installation into the snapshot dates on which it is active
    far_future = last + pd.Timedelta(days=1)
    start_idx = np.searchsorted(dates.values, ins.install_date.values, side="left")
    end_idx = np.searchsorted(dates.values, ins.removal_date.fillna(far_future).values, side="left")
    n = np.clip(end_idx - start_idx, 0, None)
    rows = np.repeat(np.arange(len(ins)), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    snap = ins.iloc[rows].reset_index(drop=True)
    snap.insert(0, "snapshot_date", dates.values[np.repeat(start_idx, n) + offs])

    # attach the removal that ended the installation
    rem_cols = rem[["serial_no", "tail_no", "position", "removal_date", "removal_type", "removal_reason"]]
    snap = snap.merge(rem_cols, on=["serial_no", "tail_no", "position", "removal_date"], how="left")

    # cycles flown between the snapshot and the removal / the end of the data
    stacked = cum.stack().rename("cum_fc")
    cum_at = lambda t, d: stacked.reindex(pd.MultiIndex.from_arrays([d, t])).to_numpy()
    fc_snap = cum_at(snap.tail_no, snap.snapshot_date)
    fc_rem = cum_at(snap.tail_no, snap.removal_date.fillna(last))
    fc_end = cum.iloc[-1].reindex(snap.tail_no).to_numpy()
    snap["fc_to_removal"] = np.where(snap.removal_date.notna(), fc_rem - fc_snap, np.nan)
    snap["days_to_removal"] = (snap.removal_date - snap.snapshot_date).dt.days
    snap["followup_fc"] = fc_end - fc_snap

    snap["label"] = ((snap.removal_type == "UNSCHEDULED") & (snap.fc_to_removal <= HORIZON_FC)).astype(int)

    # right censoring
    snap = snap[snap.followup_fc >= HORIZON_FC].reset_index(drop=True)
    return snap[KEY_COLS + ["label"] + META_COLS]


def coverage(snap: pd.DataFrame, rem: pd.DataFrame, last_date: pd.Timestamp) -> float:
    """Share of scorable unscheduled removals that have at least one positive snapshot."""
    evaluable = rem[(rem.removal_type == "UNSCHEDULED")
                    & (rem.removal_date > snap.snapshot_date.min())
                    & (rem.removal_date <= snap.snapshot_date.max())]
    hit = snap[snap.label == 1][["serial_no", "removal_date"]].drop_duplicates()
    return evaluable.merge(hit, on=["serial_no", "removal_date"], how="inner").shape[0] / len(evaluable)


def main():
    ins, rem, usage = load_raw()
    last = usage.flight_date.max()

    print("Snapshot cadence vs event coverage")
    for k in (7, 3, 1):
        s = build_snapshots(ins, rem, usage, every_days=k)
        print(f"  every {k} day(s): {len(s):>9,} rows  positive rate {s.label.mean():.2%}  "
              f"removals with >=1 positive snapshot {coverage(s, rem, last):.1%}")

    snap = build_snapshots(ins, rem, usage)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    snap.to_parquet(PROCESSED_DIR / "snapshots.parquet", index=False)

    print(f"\nSaved snapshots.parquet  (every {SNAPSHOT_EVERY_DAYS} days)")
    print(f"  rows {len(snap):,}   positives {snap.label.sum():,}   positive rate {snap.label.mean():.2%}")
    print(f"  period {snap.snapshot_date.min().date()} -> {snap.snapshot_date.max().date()}  "
          f"({snap.snapshot_date.nunique()} snapshot dates)")
    print(f"  active components per snapshot: {snap.groupby('snapshot_date').size().median():.0f} (median)")
    pos = snap[snap.label == 1]
    print(f"  positives by reason: {pos.removal_reason.value_counts().to_dict()}")
    print(f"  fc_to_removal of positives: min {pos.fc_to_removal.min():.0f}  "
          f"median {pos.fc_to_removal.median():.0f}  max {pos.fc_to_removal.max():.0f}")
    monthly = snap.groupby(snap.snapshot_date.dt.to_period("Q")).label.mean()
    print("  positive rate by quarter:", {str(k): f"{v:.2%}" for k, v in monthly.items()})


if __name__ == "__main__":
    main()
