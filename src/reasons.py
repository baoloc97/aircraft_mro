"""Plain-language reasons for one feature of one component, in maintenance terms.

Table-driven: `EXACT` maps a feature name to its sentence, `SUFFIX_RULES` covers per-channel families
(vib_/temp_/pres_ variants). A new feature gets a sentence by adding one entry; `reason()` itself never changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.features import PRIMARY_TO_CHANNEL

SENSOR_NAME = {"vib": "vibration", "temp": "temperature", "pres": "pressure"}
SENSOR_UNIT = {"vib": "ips", "temp": "°C", "pres": "psi"}


@dataclass(frozen=True)
class Context:
    feature: str
    row: pd.Series

    @property
    def v(self):
        return self.row.get(self.feature, np.nan)

    @property
    def prim(self) -> str:
        """Channel of the part's health (primary) sensor."""
        return PRIMARY_TO_CHANNEL.get(str(self.row.primary_sensor), "vib")

    @property
    def sensor(self) -> str:
        return SENSOR_NAME[self.prim].capitalize()

    @property
    def channel(self) -> str:
        """Channel named by a per-channel feature such as `pres_z_vs_ref`."""
        return self.feature.split("_")[0]


def deviation(x, what: str = "own baseline") -> str:
    if pd.isna(x):
        return f"no recent reading to compare with {what}"
    if abs(x) < 0.5:
        return f"stable (within 0.5σ of {what})"
    return f"{abs(x):.1f}σ {'above' if x >= 0 else 'below'} {what}"


def _primary_z(window_days: int, suffix: str) -> Callable[[Context], str]:
    def f(c: Context) -> str:
        raw = c.row.get(f"{c.prim}_{suffix}_vs_ref", np.nan)
        return f"{c.sensor} {deviation(raw, 'this unit' + chr(39) + 's own baseline')} over the last {window_days} days"
    return f


def _primary_slope(days: int) -> Callable[[Context], str]:
    return lambda c: f"{c.sensor} trending in the degradation direction over {days} days ({c.v:+.2f} part-σ/day)"


def _related_faults(days: int) -> Callable[[Context], str]:
    return lambda c: f"{int(c.v)} related fault message(s) (ATA {c.row.ata_chapter}) in the last {days} days"


def _troubleshooting(c: Context) -> str:
    return f"{int(c.v)} troubleshooting action(s) {'in 30 days' if '30d' in c.feature else 'since installation'}"


def _age(c: Context) -> str:
    return f"{int(c.row.csi):,} cycles / {int(c.row.days_since_install)} days since installation"


def _since_new(c: Context) -> str:
    return f"{int(c.row.csn):,} cycles since new ({c.row.csn_over_mtbur:.1f}x part MTBUR)"


def _part_type(c: Context) -> str:
    return f"Part type {c.row.part_no} (ATA {c.row.ata_chapter}) base rate"


EXACT: dict[str, Callable[[Context], str]] = {
    # health sensor, signed in the degradation direction
    "primary_z_signed": _primary_z(7, "z"),
    "primary_z3_signed": _primary_z(3, "z3"),
    "primary_slope_signed": _primary_slope(14),
    "primary_slope30_signed": _primary_slope(30),
    "primary_accel_signed": lambda c: f"{c.sensor} drift is accelerating (short-term trend steeper than the 30-day trend)",
    "primary_dev_part_signed": lambda c: f"{c.sensor} {deviation(c.v, 'the fleet baseline for this part number')}",
    "max_abs_z": lambda c: f"Largest sensor deviation {c.v:.1f}σ from own baseline",
    # fault codes
    "related_faults_3d": _related_faults(3),
    "related_faults_7d": _related_faults(7),
    "related_faults_30d": _related_faults(30),
    "related_faults_trend": lambda c: f"Related fault messages {'up' if c.v > 0 else 'down'} {abs(int(c.v))} vs the previous week",
    "caution_faults_7d": lambda c: f"{int(c.v)} CAUTION/WARNING-level related fault(s) in 7 days",
    "days_since_related_fault": lambda c: ("No related fault message in the last year" if c.v >= 365
                                           else f"Last related fault message {int(c.v)} day(s) ago"),
    "nuisance_faults_7d": lambda c: f"{int(c.v)} unrelated/nuisance fault message(s) at this position in 7 days",
    # usage and age
    "csi_over_mtbur": lambda c: f"{c.v:.0%} of part MTBUR used since installation ({int(c.row.csi):,} cycles)",
    "csi": _age,
    "days_since_install": _age,
    "csn": _since_new,
    "csn_over_mtbur": _since_new,
    "cso": lambda c: f"{int(c.v):,} cycles since last shop visit",
    "fc_per_day_30d": lambda c: f"Aircraft utilisation {c.v:.1f} cycles/day over 30 days",
    "ac_age_years": lambda c: f"Aircraft age {int(c.v)} years",
    "oat_7d": lambda c: f"Hot operating environment: {c.v:.0f}°C average outside air temperature (7 days)",
    # maintenance history
    "shop_visits": lambda c: f"{int(c.v)} shop visit(s) in the unit's life",
    "days_since_last_removal": lambda c: ("No previous removal on record" if c.v >= 730
                                          else f"Last removed {int(c.v)} days ago"),
    "prior_unscheduled_removals": lambda c: (f"Serial removed unscheduled {int(c.v)} time(s) before"
                                             + (" (possible rogue unit)" if c.v >= 3 else "")),
    "prior_nff_removals": lambda c: f"{int(c.v)} previous No-Fault-Found removal(s) of this serial",
    "troubleshooting_30d": _troubleshooting,
    "troubleshooting_install": _troubleshooting,
    "mel_deferrals_30d": lambda c: f"{int(c.v)} MEL deferral(s) in 30 days",
    # data quality
    "days_since_last_reading": lambda c: f"Last sensor reading {int(c.v)} day(s) ago",
    "sensor_missing_ratio_7d": lambda c: f"Sensor data missing on {c.v:.0%} of flying days in the last week",
    "low_history": lambda c: "Recently installed: little own history, part-number baseline used",
    # context
    "aircraft_type": lambda c: f"Aircraft type {c.v}",
    "climate_zone": lambda c: f"Operates from a {str(c.v).replace('_', ' ').lower()} base",
    "part_no": _part_type,
    "ata_chapter": _part_type,
    "primary_sensor": _part_type,
}

# Per-channel families, checked in order when there is no exact entry
SUFFIX_RULES: list[tuple[tuple[str, ...], Callable[[Context], str]]] = [
    (("_z_vs_ref", "_z3_vs_ref"), lambda c: f"{SENSOR_NAME[c.channel].capitalize()} {deviation(c.v)}"),
    (("_dev_part",), lambda c: f"{SENSOR_NAME[c.channel].capitalize()} "
                               f"{deviation(c.v, 'the part-number fleet baseline')}"),
    (("_mean_7d",), lambda c: f"{SENSOR_NAME[c.channel].capitalize()} 7-day average {c.v:,.2f} {SENSOR_UNIT[c.channel]}"),
    (("_slope_14d",), lambda c: f"{SENSOR_NAME[c.channel].capitalize()} 14-day trend {c.v:+.2f} part-σ/day"),
]


def reason(feature: str, row: pd.Series) -> str:
    """Plain-language sentence for one feature of one component (values from the untransformed row)."""
    ctx = Context(feature, row)
    if feature in EXACT:
        return EXACT[feature](ctx)
    for suffixes, render in SUFFIX_RULES:
        if feature.endswith(suffixes):
            return render(ctx)
    return f"{feature} = {ctx.v}"
