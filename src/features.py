"""Step 4 - Point-in-time feature engineering.

Every feature for a snapshot (installation, day t) uses only data dated <= t. Sensor readings and fault codes
are recorded per tail/position, so they are first attributed to the installation (serial number) that
occupied the position on that day. Data from the previous unit in the same position is never mixed in.

Feature groups
  usage / age       cycles since installation / new / overhaul, life used vs MTBUR, utilisation
  sensor            7-day level, change vs the unit's own reference window, 14-day slope, deviation from
                    the part-number baseline, missing-data indicators
  fault codes       related / nuisance / caution+ counts over 3, 7 and 30 days, days since last related fault
  maintenance       troubleshooting and MEL deferrals during this installation, serial-level removal history
  context           part number, ATA chapter, aircraft type, climate zone, aircraft age, ambient temperature

Usage:
    python -m src.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BASELINE_END, PROCESSED_DIR, RAW_DIR

SENTINELS = [0.0, -999.0, 9999.0]
CHANNELS = {"vib": "vibration_ips", "temp": "temperature_c", "pres": "pressure_psi"}
PRIMARY_TO_CHANNEL = {"vibration": "vib", "temperature": "temp", "pressure": "pres"}
# Engineering knowledge: a degrading pump or valve loses pressure; bearings and seals run hotter or rougher
DEGRADATION_SIGN = {"vib": 1, "temp": 1, "pres": -1}
SPIKE_MAD = 6.0
REF_GAP_DAYS = 15          # reference window ends 15 days before the snapshot ...
REF_WINDOW = "45D"         # ... and covers the 45 days before that

CATEGORICAL_FEATURES = ["part_no", "ata_chapter", "primary_sensor", "aircraft_type", "climate_zone"]
NUMERIC_FEATURES = [
    # usage / age
    "csi", "csn", "cso", "days_since_install", "csi_over_mtbur", "csn_over_mtbur",
    "fc_per_day_30d", "ac_age_years", "low_history",
    # sensor
    *[f"{c}_{s}" for c in CHANNELS for s in ("mean_7d", "z_vs_ref", "slope_14d", "dev_part")],
    "primary_z_signed", "primary_slope_signed", "primary_dev_part_signed", "max_abs_z",
    *[f"{c}_z3_vs_ref" for c in CHANNELS], "primary_z3_signed", "primary_slope30_signed", "primary_accel_signed",
    "sensor_missing_ratio_7d", "days_since_last_reading", "oat_7d",
    # fault codes
    "related_faults_3d", "related_faults_7d", "related_faults_30d", "related_faults_trend",
    "nuisance_faults_7d", "caution_faults_7d", "days_since_related_fault",
    # maintenance history
    "troubleshooting_30d", "troubleshooting_install", "mel_deferrals_30d",
    "prior_unscheduled_removals", "prior_nff_removals", "shop_visits", "days_since_last_removal",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# added in the second feature iteration (evaluated in Step 7 with grouped CV before being kept)
V2_FEATURES = [*[f"{c}_z3_vs_ref" for c in CHANNELS], "primary_z3_signed", "primary_slope30_signed",
               "primary_accel_signed"]


# ---------------------------------------------------------------------------- helpers
def load_raw():
    read = lambda name, **kw: pd.read_csv(RAW_DIR / f"{name}.csv", **kw)
    return dict(
        aircraft=read("aircraft"),
        parts=read("parts_catalog", dtype={"ata_chapter": str}),
        components=read("components"),
        ins=read("installations", parse_dates=["install_date", "removal_date"]),
        usage=read("flight_usage", parse_dates=["flight_date"]),
        sensors=read("sensor_readings", parse_dates=["reading_date"]),
        faults=read("fault_codes", parse_dates=["fault_date"], dtype={"ata_chapter": str}),
        maint=read("maintenance_history", parse_dates=["event_date"]),
        removals=read("removals", parse_dates=["removal_date"]),
    )


class CycleClock:
    """Cumulative flight cycles per tail at the end of any date.

    Dates before the data window are extrapolated with the tail's average daily cycles, which is how
    legacy installations without full utilisation history are handled. The average comes from the early
    baseline window only; averaging over the whole history would leak future utilisation into CSI.
    """

    def __init__(self, usage: pd.DataFrame):
        fc = usage.pivot(index="flight_date", columns="tail_no", values="flight_cycles").sort_index()
        flying = (fc > 0).astype(int)
        self.start = fc.index.min()
        self.tails = {t: i for i, t in enumerate(fc.columns)}
        self.cum = fc.cumsum().to_numpy()
        self.cum_flying = flying.cumsum().to_numpy()
        self.mean_fc = fc.loc[:BASELINE_END].mean().to_numpy()

    def _idx(self, tails, dates):
        col = np.array([self.tails[t] for t in tails])
        day = ((pd.DatetimeIndex(dates) - self.start).days).to_numpy()
        return col, day

    def at(self, tails, dates) -> np.ndarray:
        col, day = self._idx(tails, dates)
        inside = day >= 0
        out = (day + 1) * self.mean_fc[col]              # extrapolated (negative) before the window
        out[inside] = self.cum[np.minimum(day[inside], len(self.cum) - 1), col[inside]]
        return out

    def window(self, tails, dates, days: int, flying_days: bool = False) -> np.ndarray:
        """Cycles (or flying days) in the `days` days ending at each date."""
        col, day = self._idx(tails, dates)
        m = self.cum_flying if flying_days else self.cum
        end = m[np.clip(day, 0, len(m) - 1), col]
        start_day = day - days
        begin = np.where(start_day >= 0, m[np.clip(start_day, 0, len(m) - 1), col], 0)
        return (end - begin).astype(float)


def raw_as_of(raw: dict, t: pd.Timestamp) -> dict:
    """Everything the system knows at the end of day t: event tables cut at t, open installations re-opened.

    Used by the daily scoring job and by the point-in-time leakage test, so both see the same cut.
    """
    t = pd.Timestamp(t)
    cut = dict(raw)
    for name, col in (("usage", "flight_date"), ("sensors", "reading_date"), ("faults", "fault_date"),
                      ("maint", "event_date"), ("removals", "removal_date")):
        cut[name] = raw[name][raw[name][col] <= t]
    ins = raw["ins"][raw["ins"].install_date <= t].copy()
    ins.loc[ins.removal_date > t, "removal_date"] = pd.NaT
    cut["ins"] = ins
    return cut


def attribute_to_installation(df: pd.DataFrame, ins: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Assign each tail/position record to the installation occupying that position on that date."""
    right = ins[["installation_id", "tail_no", "position", "install_date", "removal_date"]].sort_values("install_date")
    out = pd.merge_asof(df.sort_values(date_col), right, left_on=date_col, right_on="install_date",
                        by=["tail_no", "position"], direction="backward")
    out = out[out.installation_id.notna()]
    return out[out.removal_date.isna() | (out[date_col] < out.removal_date)]


def windowed_counts(events: pd.DataFrame, key: str, date_col: str, snaps: pd.DataFrame,
                    snap_key: str, windows: dict[str, int | None], mask: pd.Series | None = None) -> pd.DataFrame:
    """Count events per key in (t - w, t] for each snapshot, with a searchsorted on (key, day) codes.

    A window of None means "since the beginning of time" (all events dated <= t).
    """
    ev = events if mask is None else events[mask]
    codes = {k: i for i, k in enumerate(pd.unique(pd.concat([ev[key], snaps[snap_key]])))}
    epoch = pd.Timestamp("2000-01-01")
    span = 100_000
    ev_code = np.sort(ev[key].map(codes).to_numpy() * span + (ev[date_col] - epoch).dt.days.to_numpy())
    s_key = snaps[snap_key].map(codes).to_numpy() * span
    s_day = (snaps.snapshot_date - epoch).dt.days.to_numpy()
    out = {}
    for name, w in windows.items():
        hi = np.searchsorted(ev_code, s_key + s_day, side="right")
        lo = np.searchsorted(ev_code, s_key + (s_day - w if w is not None else -1), side="right")
        out[name] = hi - lo
    return pd.DataFrame(out, index=snaps.index)


def days_since_last(events: pd.DataFrame, key: str, date_col: str, snaps: pd.DataFrame, snap_key: str,
                    cap: int = 365) -> np.ndarray:
    ev = events[[key, date_col]].rename(columns={key: snap_key, date_col: "_ev_date"}).sort_values("_ev_date")
    m = pd.merge_asof(snaps[[snap_key, "snapshot_date"]].reset_index().sort_values("snapshot_date"), ev,
                      left_on="snapshot_date", right_on="_ev_date", by=snap_key, direction="backward")
    m = m.set_index("index").reindex(snaps.index)
    return (m.snapshot_date - m._ev_date).dt.days.fillna(cap).clip(upper=cap).to_numpy()


# ---------------------------------------------------------------------------- sensor features
def clean_sensors(sensors: pd.DataFrame, ins: pd.DataFrame) -> pd.DataFrame:
    s = sensors.rename(columns={v: k for k, v in CHANNELS.items()})
    ch = list(CHANNELS)
    s[ch] = s[ch].mask(s[ch].isin(SENTINELS))
    s = attribute_to_installation(s, ins, "reading_date")
    # the reading on the installation day can belong to either unit, so skip it (initial fits are older)
    s = s[s.reading_date > s.install_date]
    s = s.sort_values(["installation_id", "reading_date"]).reset_index(drop=True)

    # single-point spikes: robust z against the unit's previous 20 readings
    g = s.groupby("installation_id")[ch]
    med = g.transform(lambda x: x.rolling(20, min_periods=5).median().shift())
    dev = (s[ch] - med).abs()
    mad = dev.groupby(s.installation_id).transform(lambda x: x.rolling(20, min_periods=5).median().shift())
    spike = dev > SPIKE_MAD * 1.4826 * mad
    s[ch] = s[ch].mask(spike)
    return s


def sensor_rolling(s: pd.DataFrame) -> pd.DataFrame:
    """Per-reading rolling statistics over the installation's own history (backward windows only)."""
    ch = list(CHANNELS)
    s = s.copy()
    s["t"] = (s.reading_date - pd.Timestamp("2024-01-01")).dt.days.astype(float)
    for c in ch:
        valid = s[c].notna()
        s[f"{c}_tt"] = s.t.where(valid)
        s[f"{c}_xt"] = s[c] * s.t
        s[f"{c}_t2"] = s[f"{c}_tt"] ** 2
    s["valid_any"] = s[ch].notna().any(axis=1).astype(float)

    def roll(window, cols, how):
        # s is sorted by (installation_id, reading_date), so the grouped result is in the same row order
        r = getattr(s.groupby("installation_id").rolling(window, on="reading_date")[cols], how)()
        return r.drop(columns="reading_date", errors="ignore").set_axis(s.index)

    m7 = roll("7D", ch, "mean").add_suffix("_mean_7d")
    valid7 = roll("7D", ["valid_any"], "sum").rename(columns={"valid_any": "valid_readings_7d"})
    ref_mean = roll(REF_WINDOW, ch, "mean").add_suffix("_ref_mean")
    ref_std = roll(REF_WINDOW, ch, "std").add_suffix("_ref_std")
    reg_cols = [f"{c}_{k}" for c in ch for k in ("tt", "xt", "t2")] + ch
    m3 = roll("3D", ch, "mean").add_suffix("_mean_3d")
    slope = pd.DataFrame(index=s.index)
    for window, suffix in (("14D", "slope_raw"), ("30D", "slope30_raw")):
        mw = roll(window, reg_cols, "mean")
        for c in ch:
            var_t = mw[f"{c}_t2"] - mw[f"{c}_tt"] ** 2
            cov = mw[f"{c}_xt"] - mw[c] * mw[f"{c}_tt"]
            slope[f"{c}_{suffix}"] = (cov / var_t).where(var_t > 1.0)

    base = s[["installation_id", "reading_date", "part_no"]]
    return pd.concat([base, m7, m3, valid7, ref_mean, ref_std, slope], axis=1)


def part_baselines(s: pd.DataFrame) -> pd.DataFrame:
    early = s[s.reading_date <= BASELINE_END]
    stats = early.groupby("part_no")[list(CHANNELS)].agg(["median", "std"])
    stats.columns = [f"{c}_part_{k}" for c, k in stats.columns]
    return stats


def sensor_features(snaps: pd.DataFrame, raw: dict) -> pd.DataFrame:
    s = clean_sensors(raw["sensors"], raw["ins"])
    s = s.merge(raw["ins"][["installation_id", "part_no"]], on="installation_id")
    roll = sensor_rolling(s)
    base = part_baselines(s)

    key = snaps[["installation_id", "snapshot_date"]].reset_index()
    cur_cols = ["installation_id", "reading_date", "valid_readings_7d",
                *[f"{c}_mean_7d" for c in CHANNELS], *[f"{c}_mean_3d" for c in CHANNELS],
                *[f"{c}_slope_raw" for c in CHANNELS], *[f"{c}_slope30_raw" for c in CHANNELS]]
    cur = pd.merge_asof(key.sort_values("snapshot_date"), roll[cur_cols].sort_values("reading_date"),
                        left_on="snapshot_date", right_on="reading_date", by="installation_id",
                        direction="backward")
    key["ref_date"] = key.snapshot_date - pd.Timedelta(days=REF_GAP_DAYS)
    ref_cols = ["installation_id", "reading_date", *[f"{c}_ref_{k}" for c in CHANNELS for k in ("mean", "std")]]
    ref = pd.merge_asof(key.sort_values("ref_date"), roll[ref_cols].sort_values("reading_date"),
                        left_on="ref_date", right_on="reading_date", by="installation_id",
                        direction="backward")
    cur = cur.set_index("index").reindex(snaps.index)
    ref = ref.set_index("index").reindex(snaps.index)

    # stale windows (the last reading is older than the window) carry no current information
    age = (snaps.snapshot_date - cur.reading_date).dt.days
    out = pd.DataFrame(index=snaps.index)
    out["days_since_last_reading"] = age.fillna(365).clip(upper=365)
    b = base.reindex(snaps.part_no).set_index(snaps.index)
    for c in CHANNELS:
        m7 = cur[f"{c}_mean_7d"].where(age <= 7)
        sd_part = b[f"{c}_part_std"]
        ref_sd = np.maximum(ref[f"{c}_ref_std"], 0.5 * sd_part)
        out[f"{c}_mean_7d"] = m7
        out[f"{c}_z_vs_ref"] = (m7 - ref[f"{c}_ref_mean"]) / ref_sd
        out[f"{c}_slope_14d"] = cur[f"{c}_slope_raw"].where(age <= 7) / sd_part   # part-sd per day
        out[f"{c}_dev_part"] = (m7 - b[f"{c}_part_median"]) / sd_part
        # short window reacts to the accelerating end of the drift, long slope to its slow start
        out[f"{c}_z3_vs_ref"] = (cur[f"{c}_mean_3d"].where(age <= 3) - ref[f"{c}_ref_mean"]) / ref_sd
        out[f"{c}_slope30"] = cur[f"{c}_slope30_raw"].where(age <= 7) / sd_part

    prim = snaps.primary_sensor.map(PRIMARY_TO_CHANNEL)
    sign = prim.map(DEGRADATION_SIGN)
    pick = lambda suffix: pd.Series(
        out[[f"{c}_{suffix}" for c in CHANNELS]].to_numpy()[np.arange(len(out)), prim.map(
            {c: i for i, c in enumerate(CHANNELS)}).to_numpy()], index=out.index)
    out["primary_z_signed"] = sign * pick("z_vs_ref")
    out["primary_slope_signed"] = sign * pick("slope_14d")
    out["primary_dev_part_signed"] = sign * pick("dev_part")
    out["primary_z3_signed"] = sign * pick("z3_vs_ref")
    out["primary_slope30_signed"] = sign * pick("slope30")
    out["primary_accel_signed"] = out.primary_slope_signed - out.primary_slope30_signed
    out = out.drop(columns=[f"{c}_slope30" for c in CHANNELS])
    out["max_abs_z"] = out[[f"{c}_z_vs_ref" for c in CHANNELS]].abs().max(axis=1, skipna=True)

    clock = raw["_clock"]
    flying7 = clock.window(snaps.tail_no, snaps.snapshot_date, 7, flying_days=True)
    valid7 = cur.valid_readings_7d.where(age <= 7).fillna(0)
    out["sensor_missing_ratio_7d"] = np.where(flying7 > 0, 1 - np.minimum(valid7 / np.maximum(flying7, 1), 1), np.nan)
    return out


# ---------------------------------------------------------------------------- other groups
def usage_features(snaps: pd.DataFrame, raw: dict) -> pd.DataFrame:
    clock = raw["_clock"]
    out = pd.DataFrame(index=snaps.index)
    csi = clock.at(snaps.tail_no, snaps.snapshot_date) - clock.at(snaps.tail_no, snaps.install_date)
    out["csi"] = np.maximum(csi, 0)
    out["csn"] = snaps.csn_at_install + out.csi
    out["cso"] = snaps.cso_at_install + out.csi
    out["days_since_install"] = (snaps.snapshot_date - snaps.install_date).dt.days
    out["csi_over_mtbur"] = out.csi / snaps.mtbur_fc
    out["csn_over_mtbur"] = out.csn / snaps.mtbur_fc
    out["fc_per_day_30d"] = clock.window(snaps.tail_no, snaps.snapshot_date, 30) / 30
    out["ac_age_years"] = snaps.snapshot_date.dt.year - snaps.manufacture_year
    out["low_history"] = (out.days_since_install < 30).astype(int)

    oat = raw["usage"].sort_values("flight_date")
    oat["oat_7d"] = oat.groupby("tail_no").avg_oat_c.transform(lambda x: x.rolling(7, min_periods=1).mean())
    m = pd.merge_asof(snaps[["tail_no", "snapshot_date"]].reset_index().sort_values("snapshot_date"),
                      oat[["tail_no", "flight_date", "oat_7d"]], left_on="snapshot_date",
                      right_on="flight_date", by="tail_no", direction="backward")
    out["oat_7d"] = m.set_index("index").reindex(snaps.index).oat_7d
    return out


def fault_features(snaps: pd.DataFrame, raw: dict) -> pd.DataFrame:
    f = attribute_to_installation(raw["faults"], raw["ins"], "fault_date")
    f = f.merge(raw["ins"][["installation_id", "part_no"]], on="installation_id")
    f = f.merge(raw["parts"][["part_no", "ata_chapter"]].rename(columns={"ata_chapter": "part_ata"}), on="part_no")
    related = f.ata_chapter == f.part_ata
    out = windowed_counts(f, "installation_id", "fault_date", snaps, "installation_id",
                          {"related_faults_3d": 3, "related_faults_7d": 7, "related_faults_30d": 30,
                           "_related_prev_7d": 14}, mask=related)
    out["_related_prev_7d"] -= out.related_faults_7d
    out["related_faults_trend"] = out.related_faults_7d - out.pop("_related_prev_7d")
    out["nuisance_faults_7d"] = windowed_counts(f, "installation_id", "fault_date", snaps, "installation_id",
                                                {"n": 7}, mask=~related)["n"]
    out["caution_faults_7d"] = windowed_counts(f, "installation_id", "fault_date", snaps, "installation_id",
                                               {"n": 7}, mask=related & (f.severity != "ADVISORY"))["n"]
    out["days_since_related_fault"] = days_since_last(f[related], "installation_id", "fault_date", snaps,
                                                      "installation_id")
    return out


def maintenance_features(snaps: pd.DataFrame, raw: dict) -> pd.DataFrame:
    m = attribute_to_installation(raw["maint"].drop(columns="serial_no"), raw["ins"], "event_date")
    ts = m.event_type == "TROUBLESHOOTING"
    out = windowed_counts(m, "installation_id", "event_date", snaps, "installation_id",
                          {"troubleshooting_30d": 30, "troubleshooting_install": None}, mask=ts)
    out["mel_deferrals_30d"] = windowed_counts(m, "installation_id", "event_date", snaps, "installation_id",
                                               {"n": 30}, mask=m.event_type == "MEL_DEFERRAL")["n"]

    # serial-level history: removals strictly before the snapshot, on any tail
    rem = raw["removals"]
    unsched = rem.removal_type == "UNSCHEDULED"
    hist = windowed_counts(rem, "serial_no", "removal_date", snaps, "serial_no",
                           {"prior_unscheduled_removals": None}, mask=unsched)
    out["prior_unscheduled_removals"] = hist.prior_unscheduled_removals
    out["prior_nff_removals"] = windowed_counts(rem, "serial_no", "removal_date", snaps, "serial_no",
                                                {"n": None}, mask=rem.removal_reason == "NO_FAULT_FOUND")["n"]
    all_rem = windowed_counts(rem, "serial_no", "removal_date", snaps, "serial_no", {"n": None})["n"]
    out["shop_visits"] = snaps.shop_visits_before_start + all_rem
    out["days_since_last_removal"] = days_since_last(rem, "serial_no", "removal_date", snaps, "serial_no",
                                                     cap=730)
    return out


# ---------------------------------------------------------------------------- main
def build_features(snaps: pd.DataFrame, raw: dict) -> pd.DataFrame:
    raw["_clock"] = CycleClock(raw["usage"])
    ctx = (snaps
           .merge(raw["ins"][["installation_id", "csn_at_install", "cso_at_install"]], on="installation_id")
           .merge(raw["parts"][["part_no", "ata_chapter", "primary_sensor", "mtbur_fc"]], on="part_no")
           .merge(raw["aircraft"][["tail_no", "aircraft_type", "climate_zone", "manufacture_year"]], on="tail_no")
           .merge(raw["components"][["serial_no", "shop_visits_before_start"]], on="serial_no"))
    assert len(ctx) == len(snaps)
    feats = pd.concat([
        usage_features(ctx, raw),
        sensor_features(ctx, raw),
        fault_features(ctx, raw),
        maintenance_features(ctx, raw),
    ], axis=1)
    out = pd.concat([ctx.drop(columns=["csn_at_install", "cso_at_install", "mtbur_fc", "manufacture_year",
                                       "shop_visits_before_start"]), feats], axis=1)
    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].astype("category")
    return out


def main():
    snaps = pd.read_parquet(PROCESSED_DIR / "snapshots.parquet")
    raw = load_raw()
    df = build_features(snaps, raw)
    missing = set(FEATURES) - set(df.columns)
    assert not missing, missing
    df.to_parquet(PROCESSED_DIR / "features.parquet", index=False)

    print(f"Saved features.parquet: {len(df):,} rows, {len(NUMERIC_FEATURES)} numeric + "
          f"{len(CATEGORICAL_FEATURES)} categorical features")
    nan = df[NUMERIC_FEATURES].isna().mean()
    print("\nFeatures with missing values:")
    print(nan[nan > 0].sort_values(ascending=False).map("{:.1%}".format).to_string())

    # sanity check on the first year only (never look at later data here)
    from sklearn.metrics import roc_auc_score
    early = df[df.snapshot_date < "2025-01-01"]
    auc = {}
    for c in NUMERIC_FEATURES:
        x = early[c].fillna(early[c].median())
        if x.nunique() > 1:
            a = roc_auc_score(early.label, x)
            auc[c] = max(a, 1 - a)
    auc = pd.Series(auc).sort_values(ascending=False)
    print("\nUnivariate AUC (2024 only, direction-free), top 15:")
    print(auc.head(15).round(3).to_string())
    print("\nWeakest 5:")
    print(auc.tail(5).round(3).to_string())


if __name__ == "__main__":
    main()
