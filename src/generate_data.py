"""Step 1 - Generate a mock MRO dataset for unscheduled component removal prediction.

Simulates a fleet day by day over two years:
  * aircraft fly a variable number of cycles per day (with AOG days and heavy checks)
  * each installed component has a Weibull time-to-failure drawn at installation
  * most failures show a precursor (sensor drift + rising fault-code rate) before removal
  * some removals are "sudden" (no precursor) or No-Fault-Found (NFF), which caps achievable recall
  * removed units go through the shop and return to the spares pool (rotables move between tails)
  * sensor data is deliberately dirty: data-link outages, sensor dropouts, random NaNs, glitches

Outputs (data/raw/*.csv):
  aircraft, parts_catalog, components, installations, flight_usage,
  sensor_readings, fault_codes, maintenance_history, removals

ACMS-style tables (sensor_readings, fault_codes) are keyed by tail_no + position only, like real
aircraft data. The serial number must be recovered by joining to installations.

Usage:
    python -m src.generate_data
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import END_DATE, HORIZON_FC, RAW_DIR, SEED, START_DATE

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
N_AIRCRAFT = 40
SPARES_RATIO = 0.15            # initial serviceable spares per part number, as a share of installed qty
SUDDEN_FAILURE_PROB = 0.08     # failures with no detectable precursor
WEAK_PRECURSOR_PROB = 0.12     # failures with a faint precursor
PRECURSOR_LEAD_FC = (25, 110)  # cycles between degradation onset and removal
ROGUE_UNIT_PROB = 0.05         # latent "rogue" units: fail early and generate NFF removals
NFF_BASE_HAZARD = 0.00012      # per-day NFF removal hazard before fault/rogue multipliers
SHOP_TAT_DAYS = (20, 60)
SCRAP_PROB = 0.10

# Data quality issues
OUTAGE_START_PROB = 0.012      # aircraft data-link outage (no sensor data for the whole tail)
OUTAGE_LEN_DAYS = (2, 10)
DROPOUT_START_PROB = 0.006     # single-sensor unit dropout
DROPOUT_LEN_DAYS = (1, 15)
RANDOM_NAN_PROB = 0.035
SPIKE_PROB = 0.002
GLITCH_PROB = 0.001

AIRCRAFT_TYPES = {
    # type: (mean daily cycles, mean block hours per cycle)
    "A320neo": (5.2, 1.7),
    "A321neo": (4.5, 2.0),
    "B737-800": (5.0, 1.8),
}

BASES = {
    # base: (climate zone, mean OAT C, seasonal amplitude C)
    "SGN": ("HOT_HUMID", 29, 2),
    "SIN": ("HOT_HUMID", 28, 1),
    "DXB": ("HOT_SANDY", 34, 8),
    "DOH": ("HOT_SANDY", 33, 8),
    "HAN": ("TEMPERATE", 24, 7),
    "NRT": ("TEMPERATE", 16, 10),
}

# Climate zone -> Weibull scale multiplier for climate-sensitive systems
CLIMATE_LIFE_FACTOR = {"HOT_HUMID": 0.85, "HOT_SANDY": 0.72, "TEMPERATE": 1.0}
CLIMATE_SENSITIVE_ATA = {"21", "36", "49"}


@dataclass(frozen=True)
class Part:
    part_no: str
    description: str
    ata: str
    positions: tuple[str, ...]
    primary_sensor: str          # which channel degrades most: vibration / temperature / pressure
    drift_sign: int              # +1 value rises when degrading, -1 value drops
    mtbur_fc: float              # mean cycles between unscheduled removals
    weibull_shape: float
    base: tuple[float, float, float]   # (vibration_ips, temperature_c, pressure_psi) nominal
    sd: tuple[float, float, float]     # day-to-day noise
    unit_cost_usd: int
    fault_codes: tuple[str, ...]
    soft_time_fc: tuple[int, int] | None = None   # wear-limit replacement window (scheduled)


PARTS: list[Part] = [
    Part("ACM-2206-01", "Air Cycle Machine", "21", ("PACK_1", "PACK_2"), "vibration", 1,
         1900, 2.2, (0.45, 85, 44), (0.04, 2.5, 1.2), 145000, ("21-51-10", "21-51-34")),
    Part("FCV-3301-02", "Pack Flow Control Valve", "21", ("PACK_1_FCV", "PACK_2_FCV"), "temperature", 1,
         1600, 1.8, (0.12, 70, 38), (0.02, 2.0, 1.0), 38000, ("21-51-41", "21-51-47")),
    Part("RCF-1104-03", "Recirculation Fan", "21", ("RECIRC_FAN_L", "RECIRC_FAN_R"), "vibration", 1,
         1700, 2.0, (0.30, 55, 15), (0.03, 1.8, 0.6), 21000, ("21-23-12",)),
    Part("IDG-7720-01", "Integrated Drive Generator", "24", ("IDG_1", "IDG_2"), "temperature", 1,
         2300, 2.4, (0.55, 110, 60), (0.04, 3.0, 1.5), 210000, ("24-21-10", "24-21-22")),
    Part("EDP-5510-04", "Engine-Driven Hydraulic Pump", "29", ("EDP_1", "EDP_2"), "pressure", -1,
         2100, 2.3, (0.38, 75, 3000), (0.03, 2.2, 25), 95000, ("29-11-15", "29-11-30")),
    Part("EHP-5620-02", "Electric Hydraulic Pump", "29", ("EHP_BLUE", "EHP_YELLOW"), "temperature", 1,
         1800, 2.0, (0.33, 80, 2950), (0.03, 2.4, 25), 62000, ("29-22-11",)),
    Part("BRK-8840-07", "Main Wheel Brake Assembly", "32", ("BRAKE_1", "BRAKE_2", "BRAKE_3", "BRAKE_4"),
         "temperature", 1, 2600, 2.8, (0.20, 180, 1500), (0.03, 8.0, 20), 48000, ("32-42-27", "32-42-31"),
         soft_time_fc=(1500, 2400)),
    Part("NWS-4410-01", "Nose Wheel Steering Actuator", "32", ("NWS_ACT",), "pressure", -1,
         2400, 2.1, (0.25, 60, 2900), (0.02, 2.0, 25), 54000, ("32-51-18",)),
    Part("PRV-6612-03", "Bleed Pressure Regulating Valve", "36", ("BLEED_PRV_1", "BLEED_PRV_2"), "pressure", -1,
         1500, 1.9, (0.18, 200, 45), (0.02, 5.0, 1.0), 41000, ("36-11-42", "36-11-45")),
    Part("PCV-6620-01", "Precooler Control Valve", "36", ("PCV_1", "PCV_2"), "temperature", 1,
         1700, 2.0, (0.15, 190, 42), (0.02, 5.0, 1.0), 36000, ("36-12-10",)),
    Part("APS-4903-02", "APU Starter Generator", "49", ("APU_SG",), "vibration", 1,
         1600, 2.2, (0.60, 95, 30), (0.05, 3.0, 1.0), 88000, ("49-41-17", "49-41-22")),
    Part("ADM-3412-05", "Air Data Module", "34", ("ADM_1", "ADM_2", "ADM_3"), "temperature", 1,
         2800, 1.6, (0.05, 45, 14.7), (0.01, 1.5, 0.3), 27000, ("34-11-19",)),
    Part("FBP-2820-03", "Fuel Boost Pump", "28", ("FUEL_PUMP_L1", "FUEL_PUMP_L2", "FUEL_PUMP_R1", "FUEL_PUMP_R2"),
         "pressure", -1, 2200, 2.1, (0.28, 50, 32), (0.03, 1.8, 0.8), 19000, ("28-21-14", "28-21-20")),
]
SENSORS = ("vibration", "temperature", "pressure")
NUISANCE_CODES = ("31-32-00", "23-73-11", "33-42-15", "45-10-00", "26-15-12", "30-21-05")
FAULT_MESSAGES = {
    "ADVISORY": "Parameter drift detected",
    "CAUTION": "Performance degraded",
    "WARNING": "System fault",
}
TROUBLESHOOT_ACTIONS = ("RESET_OK", "CONNECTOR_CLEANED", "OPS_TEST_PASSED", "ADJUSTED", "NO_FAULT_FOUND_ON_WING")


@dataclass
class Component:
    serial_no: str
    part_idx: int
    manufacture_date: pd.Timestamp
    csn: float
    tsn: float
    cso: float
    rogue: bool
    shop_visits: int
    csn_at_start: float = field(init=False)
    tsn_at_start: float = field(init=False)
    available_from: int = 0      # day index when serviceable in the spares pool
    scrapped: bool = False

    def __post_init__(self):
        self.csn_at_start = self.csn
        self.tsn_at_start = self.tsn


class FleetSimulator:
    def __init__(self, seed: int = SEED):
        self.rng = np.random.default_rng(seed)
        self.dates = pd.date_range(START_DATE, END_DATE, freq="D")
        self.n_days = len(self.dates)
        self.components: list[Component] = []
        self.serial_counter = {i: 0 for i in range(len(PARTS))}
        self.spares: dict[int, list[int]] = {i: [] for i in range(len(PARTS))}
        # event logs
        self.installations, self.removals, self.maint, self.faults = [], [], [], []
        self.sensor_chunks: list[pd.DataFrame] = []
        self.usage_rows = []

    # ------------------------------------------------------------------ setup
    def _new_serial(self, part_idx: int, day: int, fresh: bool) -> int:
        rng = self.rng
        part = PARTS[part_idx]
        self.serial_counter[part_idx] += 1
        sn = f"{part.part_no[:3]}-{self.serial_counter[part_idx]:05d}"
        if fresh:   # factory new, delivered during the simulation
            mfg = self.dates[day] - pd.Timedelta(days=int(rng.integers(30, 365)))
            csn, cso, visits = 0.0, 0.0, 0
        else:       # part of the existing rotable population at simulation start
            age_years = rng.uniform(1, 15)
            mfg = self.dates[0] - pd.Timedelta(days=int(age_years * 365))
            csn = float(rng.uniform(0.3, 1.0) * age_years * 1500)
            visits = int(rng.poisson(csn / (part.mtbur_fc * 1.5)))
            cso = float(rng.uniform(0, part.mtbur_fc * 0.5)) if visits else csn
        comp = Component(sn, part_idx, mfg, csn, csn * rng.uniform(1.6, 2.0), cso,
                         bool(rng.random() < ROGUE_UNIT_PROB), visits)
        self.components.append(comp)
        return len(self.components) - 1

    def _build_fleet(self):
        rng = self.rng
        types = list(AIRCRAFT_TYPES)
        bases = list(BASES)
        rows = []
        for i in range(N_AIRCRAFT):
            ac_type = types[i % len(types)]
            base = bases[rng.integers(len(bases))]
            mfg_year = int(rng.integers(2008, 2023))
            eis = pd.Timestamp(f"{mfg_year}-{rng.integers(1, 13):02d}-15")
            rows.append({
                "tail_no": f"AC{i + 1:03d}",
                "aircraft_type": ac_type,
                "manufacture_year": mfg_year,
                "entry_into_service": eis.date(),
                "home_base": base,
                "climate_zone": BASES[base][0],
            })
        self.aircraft = pd.DataFrame(rows)

        # one heavy check (C-check, 21 days, no flying) per aircraft per year
        self.check_days = np.zeros((N_AIRCRAFT, self.n_days), dtype=bool)
        for a in range(N_AIRCRAFT):
            for year_start in range(0, self.n_days, 365):
                start = year_start + int(rng.integers(0, 340))
                self.check_days[a, start:start + 21] = True

        # positions: one slot per (aircraft, part, position name)
        slots = []
        for a in range(N_AIRCRAFT):
            for p_idx, part in enumerate(PARTS):
                for pos in part.positions:
                    slots.append((a, p_idx, pos))
        self.slot_ac = np.array([s[0] for s in slots])
        self.slot_part = np.array([s[1] for s in slots])
        self.slot_pos = np.array([s[2] for s in slots], dtype=object)
        n = len(slots)
        self.n_slots = n

        # per-slot state for the currently installed unit
        self.slot_comp = np.full(n, -1)
        self.slot_inst_id = np.full(n, -1)
        self.csi = np.zeros(n)
        self.ttf = np.zeros(n)
        self.onset = np.zeros(n)
        self.lead = np.ones(n)
        self.sudden = np.zeros(n, dtype=bool)
        self.amp = np.zeros(n)
        self.wear_limit = np.full(n, np.inf)
        self.unit_offset = np.zeros((n, 3))
        self.fault_level = np.zeros(n)     # exponentially decayed recent related-fault count

        # spares pool
        for p_idx, part in enumerate(PARTS):
            n_spares = max(2, int(round(len(part.positions) * N_AIRCRAFT * SPARES_RATIO)))
            for _ in range(n_spares):
                self.spares[p_idx].append(self._new_serial(p_idx, 0, fresh=False))

        # initial installs: the unit is already part-way through its life
        for s in range(n):
            c = self._new_serial(self.slot_part[s], 0, fresh=False)
            self._install(s, c, day=0, initial=True)

    # ------------------------------------------------------------- lifecycle
    def _life_scale(self, slot: int, comp: Component) -> float:
        part = PARTS[self.slot_part[slot]]
        ac = self.aircraft.iloc[self.slot_ac[slot]]
        scale = part.mtbur_fc / math.gamma(1 + 1 / part.weibull_shape)
        if part.ata in CLIMATE_SENSITIVE_ATA:
            scale *= CLIMATE_LIFE_FACTOR[ac.climate_zone]
        ac_age = 2025 - ac.manufacture_year
        scale *= 1.0 - 0.012 * ac_age
        if comp.rogue:
            scale *= 0.35
        if comp.shop_visits >= 3:
            scale *= 0.85        # repeatedly repaired units degrade faster
        return scale

    def _install(self, slot: int, comp_idx: int, day: int, initial: bool = False):
        rng = self.rng
        comp = self.components[comp_idx]
        part = PARTS[comp.part_idx]
        ttf = rng.weibull(part.weibull_shape) * self._life_scale(slot, comp)
        csi0 = float(rng.uniform(0, ttf * 0.95)) if initial else 0.0
        if initial and part.soft_time_fc:
            csi0 = min(csi0, rng.uniform(*part.soft_time_fc) * rng.uniform(0, 0.95))
        if initial:
            # the unit cannot have flown more cycles since installation than in its whole life
            comp.csn = max(comp.csn, csi0 * rng.uniform(1.0, 3.0))
            comp.tsn = max(comp.tsn, comp.csn * rng.uniform(1.6, 2.0))
            comp.cso = min(max(comp.cso, csi0), comp.csn)
            comp.csn_at_start, comp.tsn_at_start = comp.csn, comp.tsn
        lead = rng.uniform(*PRECURSOR_LEAD_FC)
        r = rng.random()
        self.slot_comp[slot] = comp_idx
        self.csi[slot] = csi0
        self.ttf[slot] = ttf
        self.lead[slot] = lead
        self.onset[slot] = ttf - lead
        self.sudden[slot] = r < SUDDEN_FAILURE_PROB
        self.amp[slot] = rng.uniform(0.8, 2.0) if r < SUDDEN_FAILURE_PROB + WEAK_PRECURSOR_PROB else rng.uniform(3.0, 6.0)
        self.wear_limit[slot] = rng.uniform(*part.soft_time_fc) if part.soft_time_fc else np.inf
        if initial and csi0 >= self.wear_limit[slot]:
            self.wear_limit[slot] = csi0 + rng.uniform(50, 500)
        self.unit_offset[slot] = rng.normal(0, 0.8, 3) * np.array(part.sd)
        self.fault_level[slot] = 0.0

        inst_id = len(self.installations)
        self.slot_inst_id[slot] = inst_id
        tail = self.aircraft.tail_no.iat[self.slot_ac[slot]]
        date = self.dates[day]
        self.installations.append({
            "installation_id": f"INS{inst_id + 1:06d}",
            "serial_no": comp.serial_no,
            "part_no": part.part_no,
            "tail_no": tail,
            "position": self.slot_pos[slot],
            "install_date": date.date(),
            "removal_date": None,
            "csn_at_install": round(comp.csn - self.csi[slot]),
            "tsn_at_install": round(comp.tsn * (comp.csn - self.csi[slot]) / comp.csn, 1) if comp.csn else 0.0,
            "cso_at_install": round(max(comp.cso - self.csi[slot], 0)),
        })
        if initial:
            # the unit was fitted before the simulation window started
            fitted = date - pd.Timedelta(days=int(self.csi[slot] / 5))
            self.installations[-1]["install_date"] = fitted.date()
        else:
            self._log_maint(day, slot, comp.serial_no, "INSTALLATION", "INSTALLED_SERVICEABLE",
                            f"S/N {comp.serial_no} installed")

    def _remove(self, slot: int, day: int, removal_type: str, reason: str):
        rng = self.rng
        comp_idx = self.slot_comp[slot]
        comp = self.components[comp_idx]
        part = PARTS[comp.part_idx]
        tail = self.aircraft.tail_no.iat[self.slot_ac[slot]]
        date = self.dates[day]

        if reason == "FAILURE_CONFIRMED":
            finding = rng.choice(["BEARING_WEAR", "SEAL_LEAK", "INTERNAL_CORROSION", "ELECTRICAL_FAULT",
                                  "CONTAMINATION"])
        elif reason == "NO_FAULT_FOUND":
            finding = "NO_FAULT_FOUND"
        else:
            finding = "ROUTINE_OVERHAUL"
        self.removals.append({
            "removal_id": f"REM{len(self.removals) + 1:06d}",
            "removal_date": date.date(),
            "serial_no": comp.serial_no,
            "part_no": part.part_no,
            "tail_no": tail,
            "position": self.slot_pos[slot],
            "removal_type": removal_type,
            "removal_reason": reason,
            "csn_at_removal": round(comp.csn),
            "csi_at_removal": round(self.csi[slot]),
            "shop_finding": finding,          # only known after the shop visit (post-event field)
        })
        self.installations[self.slot_inst_id[slot]]["removal_date"] = date.date()
        self._log_maint(day, slot, comp.serial_no, "REMOVAL", f"{removal_type}_{reason}",
                        f"S/N {comp.serial_no} removed")

        # send to shop -> back to the spares pool (or scrap)
        comp.shop_visits += 1
        comp.cso = 0.0
        if reason == "NO_FAULT_FOUND":
            comp.available_from = day + int(rng.integers(7, 20))
        else:
            comp.available_from = day + int(rng.integers(*SHOP_TAT_DAYS))
        comp.scrapped = rng.random() < SCRAP_PROB and reason != "NO_FAULT_FOUND"
        if not comp.scrapped:
            self.spares[comp.part_idx].append(comp_idx)

        # fit a replacement: serviceable spare if available, otherwise factory new
        pool = [c for c in self.spares[comp.part_idx]
                if self.components[c].available_from <= day and c != comp_idx]
        if pool:
            new_c = pool[int(rng.integers(len(pool)))]
            self.spares[comp.part_idx].remove(new_c)
        else:
            new_c = self._new_serial(comp.part_idx, day, fresh=True)
        self._install(slot, new_c, day)

    def _log_maint(self, day, slot, serial, event_type, action, notes):
        self.maint.append({
            "event_id": f"EVT{len(self.maint) + 1:07d}",
            "event_date": self.dates[day].date(),
            "tail_no": self.aircraft.tail_no.iat[self.slot_ac[slot]],
            "position": self.slot_pos[slot],
            "serial_no": serial,
            "event_type": event_type,
            "action": action,
            "notes": notes,
        })

    # ---------------------------------------------------------------- daily loop
    def run(self):
        rng = self.rng
        self._build_fleet()
        ac = self.aircraft
        type_cycles = np.array([AIRCRAFT_TYPES[t][0] for t in ac.aircraft_type])
        type_hours = np.array([AIRCRAFT_TYPES[t][1] for t in ac.aircraft_type])
        base_info = [BASES[b] for b in ac.home_base]
        oat_mean = np.array([b[1] for b in base_info])
        oat_amp = np.array([b[2] for b in base_info])
        outage_left = np.zeros(N_AIRCRAFT, dtype=int)
        dropout_left = np.zeros(self.n_slots, dtype=int)
        tails = ac.tail_no.to_numpy()
        part_base = np.array([p.base for p in PARTS])
        part_sd = np.array([p.sd for p in PARTS])
        primary_idx = np.array([SENSORS.index(p.primary_sensor) for p in PARTS])
        secondary_idx = (primary_idx + 1) % 3
        drift_sign = np.array([p.drift_sign for p in PARTS])
        part_mtbur = np.array([p.mtbur_fc for p in PARTS])

        for d in range(self.n_days):
            date = self.dates[d]
            doy = date.dayofyear

            # --- flying
            in_check = self.check_days[:, d]
            aog = rng.random(N_AIRCRAFT) < 0.015
            fc = rng.poisson(type_cycles)
            fc[in_check | aog] = 0
            fh = np.round(fc * type_hours * rng.uniform(0.9, 1.1, N_AIRCRAFT), 1)
            oat = oat_mean + oat_amp * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 1.5, N_AIRCRAFT)
            for a in range(N_AIRCRAFT):
                self.usage_rows.append((date.date(), tails[a], int(fc[a]), float(fh[a]), round(float(oat[a]), 1),
                                        bool(in_check[a])))

            # --- heavy check start: opportunistic scheduled removals of high-time units
            check_start = in_check & ~(self.check_days[:, d - 1] if d > 0 else np.zeros(N_AIRCRAFT, bool))
            if check_start.any():
                for s in np.where(check_start[self.slot_ac])[0]:
                    if self.csi[s] > 0.8 * part_mtbur[self.slot_part[s]] and rng.random() < 0.3:
                        self._remove(s, d, "SCHEDULED", "OPPORTUNISTIC_CHECK")

            # --- accumulate usage
            slot_fc = fc[self.slot_ac].astype(float)
            self.csi += slot_fc
            for s in np.where(slot_fc > 0)[0]:
                comp = self.components[self.slot_comp[s]]
                comp.csn += slot_fc[s]
                comp.tsn += fh[self.slot_ac[s]] * slot_fc[s] / max(fc[self.slot_ac[s]], 1)
                comp.cso += slot_fc[s]

            # --- degradation progress in [0, 1]
            progress = np.clip((self.csi - self.onset) / self.lead, 0, 1)
            progress[self.sudden] = 0.0
            p_idx = self.slot_part
            rogue = np.array([self.components[c].rogue for c in self.slot_comp])

            # --- fault codes (related to the installed unit)
            flying = slot_fc > 0
            related_rate = (0.004 + 0.55 * progress ** 2 + 0.02 * rogue) * flying
            n_related = rng.poisson(related_rate)
            self.fault_level = self.fault_level * 0.8 + n_related
            for s in np.where(n_related > 0)[0]:
                part = PARTS[p_idx[s]]
                for _ in range(n_related[s]):
                    sev = "WARNING" if progress[s] > 0.85 and rng.random() < 0.4 else \
                          "CAUTION" if progress[s] > 0.5 and rng.random() < 0.6 else "ADVISORY"
                    code = part.fault_codes[int(rng.integers(len(part.fault_codes)))]
                    self._log_fault(d, s, code, part.ata, sev, f"{part.description}: {FAULT_MESSAGES[sev]}")
                    if rng.random() < 0.3:
                        self._log_maint(d, s, self.components[self.slot_comp[s]].serial_no, "TROUBLESHOOTING",
                                        TROUBLESHOOT_ACTIONS[int(rng.integers(len(TROUBLESHOOT_ACTIONS)))],
                                        f"Troubleshooting after {code}")
                    if sev != "ADVISORY" and rng.random() < 0.15:
                        self._log_maint(d, s, self.components[self.slot_comp[s]].serial_no, "MEL_DEFERRAL",
                                        "DEFERRED_CAT_C", f"Item deferred under MEL after {code}")

            # --- nuisance fault codes, unrelated to component health
            n_nuis = rng.poisson(0.35 * (fc > 0))
            for a in np.where(n_nuis > 0)[0]:
                for _ in range(n_nuis[a]):
                    s = int(rng.choice(np.where(self.slot_ac == a)[0]))
                    code = NUISANCE_CODES[int(rng.integers(len(NUISANCE_CODES)))]
                    self._log_fault(d, s, code, code[:2], "ADVISORY", "Spurious/intermittent message")

            # --- sensor readings (only on flying days, subject to outages)
            new_out = (rng.random(N_AIRCRAFT) < OUTAGE_START_PROB) & (outage_left == 0)
            outage_left[new_out] = rng.integers(*OUTAGE_LEN_DAYS, new_out.sum())
            new_drop = (rng.random(self.n_slots) < DROPOUT_START_PROB) & (dropout_left == 0)
            dropout_left[new_drop] = rng.integers(*DROPOUT_LEN_DAYS, new_drop.sum())

            report = flying & (outage_left[self.slot_ac] == 0)
            idx = np.where(report)[0]
            if len(idx):
                base = part_base[p_idx[idx]]
                sd = part_sd[p_idx[idx]]
                vals = base + self.unit_offset[idx] + rng.normal(0, 1, (len(idx), 3)) * sd
                # slow ageing drift with cycles since installation
                age_frac = np.clip(self.csi[idx] / part_mtbur[p_idx[idx]], 0, 2)
                sign = drift_sign[p_idx[idx]]
                pri = primary_idx[p_idx[idx]]
                sec = secondary_idx[p_idx[idx]]
                rows = np.arange(len(idx))
                vals[rows, pri] += sign * 0.6 * age_frac * sd[rows, pri]
                # precursor drift ahead of failure
                prog = progress[idx]
                vals[rows, pri] += sign * self.amp[idx] * prog ** 1.5 * sd[rows, pri]
                vals[rows, sec] += 0.4 * self.amp[idx] * prog ** 2 * sd[rows, sec]
                # ambient temperature effect on temperature channel
                vals[:, 1] += 0.35 * (oat[self.slot_ac[idx]] - 25)
                # rogue units run noisier
                vals += rogue[idx, None] * rng.normal(0, 1.0, (len(idx), 3)) * sd

                # data quality issues
                vals[dropout_left[idx] > 0, :] = np.nan
                vals[rng.random(vals.shape) < RANDOM_NAN_PROB] = np.nan
                spike = rng.random(vals.shape) < SPIKE_PROB
                vals[spike] *= rng.uniform(2.5, 6.0, spike.sum())
                glitch = rng.random(vals.shape) < GLITCH_PROB
                vals[glitch] = rng.choice([0.0, -999.0, 9999.0], glitch.sum())

                self.sensor_chunks.append(pd.DataFrame({
                    "reading_date": date.date(),
                    "tail_no": tails[self.slot_ac[idx]],
                    "position": self.slot_pos[idx],
                    "vibration_ips": np.round(vals[:, 0], 3),
                    "temperature_c": np.round(vals[:, 1], 1),
                    "pressure_psi": np.round(vals[:, 2], 1),
                }))
            outage_left = np.maximum(outage_left - 1, 0)
            dropout_left = np.maximum(dropout_left - 1, 0)

            # --- removals
            for s in np.where(self.csi >= self.ttf)[0]:
                self._remove(s, d, "UNSCHEDULED", "FAILURE_CONFIRMED")
            nff_h = NFF_BASE_HAZARD * (1 + 5 * rogue) * (1 + 1.5 * self.fault_level) * flying
            for s in np.where(rng.random(self.n_slots) < nff_h)[0]:
                self._remove(s, d, "UNSCHEDULED", "NO_FAULT_FOUND")
            for s in np.where(self.csi >= self.wear_limit)[0]:
                self._remove(s, d, "SCHEDULED", "WEAR_LIMIT")

    def _log_fault(self, day, slot, code, ata, severity, message):
        self.faults.append((len(self.faults) + 1, self.dates[day].date(),
                            self.aircraft.tail_no.iat[self.slot_ac[slot]], self.slot_pos[slot],
                            code, ata, severity, message))

    # ---------------------------------------------------------------- outputs
    def tables(self) -> dict[str, pd.DataFrame]:
        parts = pd.DataFrame([{
            "part_no": p.part_no,
            "part_description": p.description,
            "ata_chapter": p.ata,
            "qty_per_aircraft": len(p.positions),
            "primary_sensor": p.primary_sensor,
            "mtbur_fc": p.mtbur_fc,
            "soft_time_fc": p.soft_time_fc[0] if p.soft_time_fc else None,
            "unit_cost_usd": p.unit_cost_usd,
        } for p in PARTS])
        components = pd.DataFrame([{
            "serial_no": c.serial_no,
            "part_no": PARTS[c.part_idx].part_no,
            "manufacture_date": c.manufacture_date.date(),
            "csn_at_start": round(c.csn_at_start),
            "tsn_at_start": round(c.tsn_at_start, 1),
            "shop_visits_before_start": c.shop_visits - sum(
                1 for r in self.removals if r["serial_no"] == c.serial_no),
        } for c in self.components])
        usage = pd.DataFrame(self.usage_rows, columns=[
            "flight_date", "tail_no", "flight_cycles", "flight_hours", "avg_oat_c", "in_heavy_check"])
        faults = pd.DataFrame(self.faults, columns=[
            "fault_id", "fault_date", "tail_no", "position", "fault_code", "ata_chapter", "severity", "message"])
        faults["fault_id"] = faults.fault_id.map(lambda i: f"FLT{i:07d}")
        return {
            "aircraft": self.aircraft,
            "parts_catalog": parts,
            "components": components,
            "installations": pd.DataFrame(self.installations),
            "flight_usage": usage,
            "sensor_readings": pd.concat(self.sensor_chunks, ignore_index=True),
            "fault_codes": faults,
            "maintenance_history": pd.DataFrame(self.maint),
            "removals": pd.DataFrame(self.removals),
        }


def approx_positive_rate(tables: dict[str, pd.DataFrame]) -> float:
    """Share of installed component-days with an unscheduled removal in the next HORIZON_FC cycles."""
    usage = tables["flight_usage"].sort_values(["tail_no", "flight_date"])
    usage["cum_fc"] = usage.groupby("tail_no").flight_cycles.cumsum()
    cum = {t: g.set_index("flight_date").cum_fc for t, g in usage.groupby("tail_no")}
    n_days = usage.flight_date.nunique()
    n_slots = sum(len(p.positions) for p in PARTS) * N_AIRCRAFT
    positives = 0
    for r in tables["removals"].query("removal_type == 'UNSCHEDULED'").itertuples():
        c = cum[r.tail_no]
        # snapshots strictly before the removal day whose remaining cycles to removal <= horizon
        before = c[c.index < r.removal_date]
        positives += int(((c[r.removal_date] - before) <= HORIZON_FC).sum())
    return positives / (n_days * n_slots)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sim = FleetSimulator()
    sim.run()
    tables = sim.tables()
    for name, df in tables.items():
        df.to_csv(RAW_DIR / f"{name}.csv", index=False)

    print("Rows per table")
    for name, df in tables.items():
        print(f"  {name:<20} {len(df):>10,}")
    rem = tables["removals"]
    print("\nRemovals by type / reason")
    print(rem.groupby(["removal_type", "removal_reason"]).size().to_string())
    sens = tables["sensor_readings"][["vibration_ips", "temperature_c", "pressure_psi"]]
    print(f"\nSensor missing rate: {sens.isna().mean().mean():.1%}")
    print(f"Distinct serials: {tables['components'].serial_no.nunique():,}  "
          f"(serials that moved between tails: "
          f"{(tables['installations'].groupby('serial_no').tail_no.nunique() > 1).sum():,})")
    print(f"Approx. positive rate (component-day level): {approx_positive_rate(tables):.2%}")


if __name__ == "__main__":
    main()
