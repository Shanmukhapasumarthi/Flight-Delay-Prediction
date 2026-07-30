"""
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.config import MODELS_DIR
from src.models.explain import explain_rows, get_explainer, narrate
from src.models.featurize_online import UnknownEntity, featurize, load_store, to_frame

STATE: dict[str, Any] = {}
STARTED = time.time()

RISK_BANDS = [(0.15, "low"), (0.30, "moderate"), (0.50, "elevated"), (1.01, "high")]


def band(p: float) -> str:
    return next(name for edge, name in RISK_BANDS if p < edge)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(MODELS_DIR / "final_model.joblib")
    STATE["bundle"] = bundle
    STATE["store"] = load_store()
    STATE["explainer"] = get_explainer(bundle)
    thr_path = MODELS_DIR / "operating_threshold.json"
    STATE["threshold"] = (json.loads(thr_path.read_text())["threshold"]
                          if thr_path.exists() else bundle["threshold_f1"])
    yield
    STATE.clear()


app = FastAPI(
    title="Flight Delay Risk API",
    version="1.0.0",
    description="Calibrated pre-departure delay risk with per-flight explanations.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------
class WeatherObs(BaseModel):
    """Forecast valid at the scheduled hour. Any field may be omitted."""
    temperature_c: float | None = None
    dewpoint_c: float | None = None
    wind_speed_kt: float | None = Field(None, ge=0, le=150)
    wind_gust_kt: float | None = Field(None, ge=0, le=200)
    precip_mm: float | None = Field(None, ge=0, le=200)
    snow_mm: float | None = Field(None, ge=0, le=500)
    visibility_km: float | None = Field(None, ge=0, le=20)
    cloud_ceiling_ft: float | None = Field(None, ge=0, le=50000)
    condition: str | None = None


class FlightRequest(BaseModel):
    carrier_code: str = Field(..., min_length=2, max_length=3, examples=["AA"])
    origin: str = Field(..., min_length=3, max_length=3, examples=["ORD"])
    destination: str = Field(..., min_length=3, max_length=3, examples=["ATL"])
    scheduled_departure_local: str = Field(..., examples=["2024-04-12T17:45:00"])
    scheduled_arrival_local: str | None = None
    flight_number: int | None = None
    tail_number: str | None = None
    aircraft_model: str | None = None
    aircraft_age_years: float | None = Field(None, ge=0, le=50)
    seats: int | None = Field(None, ge=1, le=900)
    scheduled_turnaround_min: float | None = Field(None, ge=0, le=1440)
    scheduled_ground_time_min: float | None = Field(None, ge=0, le=1440)
    prev_leg_arr_delay_min: float | None = Field(
        None, description="Actual arrival delay of the inbound leg, if it has landed")
    prev_leg_dep_delay_min: float | None = Field(
        None, description="Departure delay of the inbound leg, if still airborne")
    aircraft_cum_delay_today: float | None = None
    origin_weather: WeatherObs | None = None
    dest_weather: WeatherObs | None = None

    @field_validator("carrier_code", "origin", "destination")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.strip().upper()


class BatchRequest(BaseModel):
    flights: list[FlightRequest] = Field(..., min_length=1, max_length=500)
    explain: bool = True
    top_n_reasons: int = Field(4, ge=1, le=10)
    threshold: float | None = Field(None, ge=0, le=1,
                                    description="Override the operating threshold")


class Contribution(BaseModel):
    feature: str
    label: str
    display_value: str
    shap_value: float
    direction: str


class Prediction(BaseModel):
    flight: str
    route: str
    scheduled_departure_local: str
    delay_probability: float
    risk_band: str
    threshold: float
    recommend_intervention: bool
    expected_delay_minutes: float | None = None
    top_factors: list[Contribution] = []
    explanation: str | None = None


# --------------------------------------------------------------------------
def _prepare(reqs: list[FlightRequest]) -> pd.DataFrame:
    bundle = STATE["bundle"]
    rows = []
    for r in reqs:
        payload = r.model_dump()
        for k in ("origin_weather", "dest_weather"):
            if payload.get(k):
                payload[k] = {kk: vv for kk, vv in payload[k].items() if vv is not None}
        try:
            rows.append(featurize(payload, STATE["store"]))
        except UnknownEntity as e:
            raise HTTPException(status_code=422, detail=str(e))
    return to_frame(rows, bundle["features"], bundle["categories"])


def _predict(X: pd.DataFrame) -> np.ndarray:
    return STATE["bundle"]["calibrator"].predict_proba(X)[:, 1]


def _assemble(reqs, X, proba, explain: bool, top_n: int, thr: float) -> list[Prediction]:
    contribs = explain_rows(STATE["bundle"], X, top_n=top_n,
                            explainer=STATE["explainer"]) if explain else [[]] * len(reqs)
    avg_delay = 44.0
    out = []
    for r, p, c in zip(reqs, proba, contribs):
        fl = f"{r.carrier_code}{r.flight_number}" if r.flight_number else r.carrier_code
        out.append(Prediction(
            flight=fl,
            route=f"{r.origin}-{r.destination}",
            scheduled_departure_local=r.scheduled_departure_local,
            delay_probability=round(float(p), 4),
            risk_band=band(float(p)),
            threshold=thr,
            recommend_intervention=bool(p >= thr),
            expected_delay_minutes=round(float(p) * avg_delay, 1),
            top_factors=[Contribution(**{k: v for k, v in x.items()
                                         if k in Contribution.model_fields}) for x in c],
            explanation=narrate(c, float(p)) if explain and c else None,
        ))
    return out


# --------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
def health() -> dict:
    b = STATE.get("bundle")
    return {
        "status": "ok" if b else "model_not_loaded",
        "uptime_seconds": round(time.time() - STARTED, 1),
        "model_type": b["model_type"] if b else None,
        "trained_at": b["trained_at"] if b else None,
        "n_features": len(b["features"]) if b else 0,
        "feature_store_as_of": STATE.get("store", {}).get("as_of"),
        "operating_threshold": STATE.get("threshold"),
    }


@app.get("/model-info", tags=["ops"])
def model_info() -> dict:
    b = STATE["bundle"]
    return {
        "model_type": b["model_type"],
        "trained_at": b["trained_at"],
        "train_rows": b["train_rows"],
        "hyperparameters": b["params"],
        "test_metrics": b["metrics_test"],
        "features": b["features"],
        "operating_threshold": STATE["threshold"],
        "threshold_note": ("chosen by expected-value optimisation over intervention costs, "
                           "not by F1 or 0.5"),
    }


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict(req: FlightRequest, explain: bool = True) -> Prediction:
    X = _prepare([req])
    p = _predict(X)
    return _assemble([req], X, p, explain, 4, STATE["threshold"])[0]


@app.post("/delay-risk", tags=["prediction"])
def delay_risk(req: BatchRequest) -> dict:
    thr = req.threshold if req.threshold is not None else STATE["threshold"]
    X = _prepare(req.flights)
    proba = _predict(X)
    preds = _assemble(req.flights, X, proba, req.explain, req.top_n_reasons, thr)
    preds.sort(key=lambda p: -p.delay_probability)
    flagged = [p for p in preds if p.recommend_intervention]
    return {
        "n_flights": len(preds),
        "threshold": thr,
        "n_flagged": len(flagged),
        "mean_risk": round(float(np.mean(proba)), 4),
        "expected_delayed_flights": round(float(np.sum(proba)), 1),
        "watchlist": [p.model_dump() for p in flagged],
        "all": [p.model_dump() for p in preds],
    }
