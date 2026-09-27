"""FastAPI scoring service (PoC).

Endpoints
  GET  /health                         model version, decision rule, thresholds
  GET  /fleet/risk?date=...&level=...  ranked risk list for the whole fleet as of a date (daily job view)
  POST /predict                        risk for specific components as of a date, with reasons

Run:
    uvicorn api.main:app --reload
"""
from __future__ import annotations

from datetime import date as Date
from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.predict import load_scorer, score_fleet

app = FastAPI(title="Unscheduled Removal Risk API", version="1.0.0",
              description="Risk of an unscheduled component removal within the next 30 flight cycles.")

DATA_START, DATA_END = pd.Timestamp("2024-02-01"), pd.Timestamp("2025-12-31")


class Factor(BaseModel):
    feature: str
    value: float | str | None
    impact: float = Field(description="SHAP contribution to the log-odds of removal")
    reason: str


class DataQuality(BaseModel):
    low_history: bool = Field(description="Installed < 30 days ago: little own history, part baseline used")
    sensor_missing_ratio_7d: float | None
    days_since_last_reading: float | None
    confidence: Literal["normal", "reduced"]


class ComponentRisk(BaseModel):
    as_of_date: Date
    serial_no: str
    part_no: str
    part_description: str | None = None
    ata_chapter: str
    tail_no: str
    position: str
    risk_score: float = Field(description="Calibrated probability of an unscheduled removal within 30 FC")
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]
    recommended_action: str
    top_factors: list[Factor]
    data_quality: DataQuality
    model_version: str


class FleetRisk(BaseModel):
    as_of_date: Date
    active_components: int
    alerts: int
    alert_rate: float
    returned: int
    components: list[ComponentRisk]


class PredictItem(BaseModel):
    serial_no: str | None = None
    tail_no: str | None = None
    position: str | None = None


class PredictRequest(BaseModel):
    as_of_date: Date
    components: list[PredictItem]


@lru_cache(maxsize=16)
def _scored(day: str) -> pd.DataFrame:
    return score_fleet(day)


@lru_cache(maxsize=1)
def _descriptions() -> dict:
    from src.config import RAW_DIR
    p = pd.read_csv(RAW_DIR / "parts_catalog.csv")
    return dict(zip(p.part_no, p.part_description))


def _check_date(d: Date) -> str:
    t = pd.Timestamp(d)
    if not DATA_START <= t <= DATA_END:
        raise HTTPException(422, f"as_of_date must be between {DATA_START.date()} and {DATA_END.date()} "
                                 "(the mock data window)")
    return str(t.date())


def _to_item(r: pd.Series) -> ComponentRisk:
    scorer = load_scorer()
    missing = None if pd.isna(r.sensor_missing_ratio_7d) else round(float(r.sensor_missing_ratio_7d), 2)
    reduced = bool(r.low_history) or (missing is not None and missing > 0.5) or r.days_since_last_reading > 7
    return ComponentRisk(
        as_of_date=r.snapshot_date.date(), serial_no=r.serial_no, part_no=r.part_no,
        part_description=_descriptions().get(r.part_no), ata_chapter=str(r.ata_chapter), tail_no=r.tail_no,
        position=r.position, risk_score=round(float(r.risk_score), 4), risk_level=r.risk_level,
        recommended_action=r.recommended_action,
        top_factors=[Factor(**{**f, "value": round(f["value"], 3) if isinstance(f["value"], float) else f["value"]})
                     for f in r.top_factors],
        data_quality=DataQuality(low_history=bool(r.low_history), sensor_missing_ratio_7d=missing,
                                 days_since_last_reading=float(r.days_since_last_reading),
                                 confidence="reduced" if reduced else "normal"),
        model_version=scorer.version,
    )


@app.get("/health")
def health():
    s = load_scorer()
    return {"status": "ok", "model_version": s.version, "scorer": s.metadata.get("decision"),
            "models": list(s.models), "alert_cap_per_run": s.max_alert_rate,
            "flag_threshold_risk": round(s.metadata.get("calibrated_threshold", float("nan")), 4),
            "high_threshold_risk": round(s.metadata.get("calibrated_high_threshold", float("nan")), 4),
            "train_period": s.metadata.get("train_period"), "horizon": "30 flight cycles"}


@app.get("/fleet/risk", response_model=FleetRisk)
def fleet_risk(date: Date = Query(..., description="Scoring date, YYYY-MM-DD"),
               level: Literal["HIGH", "MEDIUM", "LOW", "ALERTS", "ALL"] = "ALERTS",
               limit: int = Query(20, ge=1, le=2000)):
    res = _scored(_check_date(date))
    sel = {"ALL": res, "ALERTS": res[res.flag]}.get(level, res[res.risk_level == level])
    return FleetRisk(as_of_date=date, active_components=len(res), alerts=int(res.flag.sum()),
                     alert_rate=round(float(res.flag.mean()), 4), returned=min(limit, len(sel)),
                     components=[_to_item(r) for _, r in sel.head(limit).iterrows()])


@app.post("/predict", response_model=list[ComponentRisk])
def predict(req: PredictRequest):
    res = _scored(_check_date(req.as_of_date))
    out = []
    for item in req.components:
        if item.serial_no:
            hit = res[res.serial_no == item.serial_no]
        elif item.tail_no and item.position:
            hit = res[(res.tail_no == item.tail_no) & (res.position == item.position)]
        else:
            raise HTTPException(422, "each component needs serial_no, or tail_no + position")
        if hit.empty:
            raise HTTPException(404, f"component {item.model_dump(exclude_none=True)} is not installed on "
                                     f"{req.as_of_date}")
        out.append(_to_item(hit.iloc[0]))
    return out
