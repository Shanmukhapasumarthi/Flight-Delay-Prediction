"""Online featurizer - the serving-time twin of `build_features.py`.

Given a compact description of one future flight plus (optionally) a weather
forecast and the state of the inbound aircraft, this reconstructs the exact
feature vector the model was trained on, filling any gap from the feature
store rather than guessing.

Anything the caller omits degrades gracefully:
    weather        -> airport/month climatological normals
    aircraft       -> tail-number lookup, then carrier-typical medians
    upstream leg   -> assumed clean (0 min) with prev_leg_known = 0
    congestion     -> typical demand for that airport/weekday/hour
"""
from __future__ import annotations

import datetime as dt
import functools
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from src.config import MODELS_DIR

EARTH_R_KM = 6371.0
TIME_BLOCK_BINS = [(-1, 5, "night"), (5, 9, "early_morning"), (9, 12, "late_morning"),
                   (12, 16, "afternoon"), (16, 20, "evening"), (20, 24, "late_evening")]


@functools.lru_cache(maxsize=1)
def load_store(path: str | None = None) -> dict:
    p = path or (MODELS_DIR / "feature_store.json")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=8)
def _holiday_set(year: int) -> tuple[set, set]:
    import holidays as hol
    us = hol.UnitedStates(years=[year - 1, year, year + 1], observed=True)
    heavy = ("Thanksgiving", "Christmas", "New Year", "Independence Day",
             "Memorial Day", "Labor Day")
    major = {d for d, n in us.items() if any(h in n for h in heavy)}
    for y in (year - 1, year, year + 1):
        major |= {dt.date(y, 12, 24), dt.date(y, 12, 31), dt.date(y, 12, 26)}
    return set(us.keys()), major


def haversine(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(min(1.0, a)))


def bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def wx_severity(precip, gust, vis, snow, storm, ceiling) -> float:
    return float(
        0.85 * np.clip(precip / 6.0, 0, 1.7)
        + 1.25 * np.clip((gust - 24) / 26.0, 0, 1.6)
        + 1.05 * np.clip((5.0 - vis) / 5.0, 0, 1.0)
        + 1.55 * np.clip(snow / 18.0, 0, 1.7)
        + 1.15 * float(storm)
        + 0.35 * float(ceiling < 800))


def _wx(block: dict | None, normals: dict, condition_default="Clear") -> dict:
    b = dict(block or {})
    out = {
        "temperature_c": b.get("temperature_c", normals.get("dep_wx_temperature_c", 15.0)),
        "wind_speed_kt": b.get("wind_speed_kt", normals.get("dep_wx_wind_speed_kt", 8.0)),
        "wind_gust_kt": b.get("wind_gust_kt", normals.get("dep_wx_wind_gust_kt", 10.0)),
        "precip_mm": b.get("precip_mm", normals.get("dep_wx_precip_mm", 0.0)),
        "snow_mm": b.get("snow_mm", 0.0),
        "visibility_km": b.get("visibility_km", normals.get("dep_wx_visibility_km", 16.0)),
        "cloud_ceiling_ft": b.get("cloud_ceiling_ft", normals.get("dep_wx_cloud_ceiling_ft", 12000.0)),
        "condition": b.get("condition", condition_default),
    }
    out["dewpoint_c"] = b.get("dewpoint_c", out["temperature_c"] - 4.0)
    out["is_storm"] = float(out["condition"] == "Thunderstorm"
                            or (out["wind_gust_kt"] > 35 and out["precip_mm"] > 3))
    return out


class UnknownEntity(ValueError):
    pass


def featurize(req: dict[str, Any], store: dict | None = None) -> dict[str, Any]:
    """Build the full feature dict for one flight request."""
    s = store or load_store()
    d = s["defaults"]
    ap = s["airports"]

    origin = str(req["origin"]).strip().upper()
    dest = str(req["destination"]).strip().upper()
    carrier = str(req["carrier_code"]).strip().upper()
    if origin not in ap:
        raise UnknownEntity(f"unknown origin airport '{origin}'")
    if dest not in ap:
        raise UnknownEntity(f"unknown destination airport '{dest}'")
    if origin == dest:
        raise UnknownEntity("origin and destination must differ")

    o, t = ap[origin], ap[dest]
    dep = pd.Timestamp(req["scheduled_departure_local"])
    if pd.isna(dep):
        raise UnknownEntity("scheduled_departure_local could not be parsed")

    f: dict[str, Any] = {}

    # ---------------- time ------------------------------------------------
    hour_dec = dep.hour + dep.minute / 60
    f["dep_time_decimal"] = float(hour_dec)
    f["dep_minute"] = int(dep.minute)
    f["day_of_week"] = int(dep.dayofweek)
    f["hour_sin"] = math.sin(2 * math.pi * hour_dec / 24)
    f["hour_cos"] = math.cos(2 * math.pi * hour_dec / 24)
    f["doy_sin"] = math.sin(2 * math.pi * dep.dayofyear / 365.25)
    f["doy_cos"] = math.cos(2 * math.pi * dep.dayofyear / 365.25)
    f["time_block"] = next(n for lo, hi, n in TIME_BLOCK_BINS if lo < dep.hour <= hi)

    _, major = _holiday_set(dep.year)
    if major:
        gap = min(abs((dep.date() - h).days) for h in major)
        f["holiday_proximity"] = float(np.clip(1 - gap / 4.0, 0, 1))
    else:
        f["holiday_proximity"] = 0.0

    # ---------------- geography & flight ---------------------------------
    dist = haversine(o["latitude"], o["longitude"], t["latitude"], t["longitude"])
    brg = bearing(o["latitude"], o["longitude"], t["latitude"], t["longitude"])
    f["lat_delta"] = float(t["latitude"] - o["latitude"])
    f["bearing_sin"] = math.sin(math.radians(brg))
    f["bearing_cos"] = math.cos(math.radians(brg))

    route_key = f"{origin}-{dest}"
    route = s["route"].get(route_key, {})
    if req.get("scheduled_arrival_local"):
        arr = pd.Timestamp(req["scheduled_arrival_local"])
        block = (arr - dep).total_seconds() / 60
        # local-time arithmetic across zones would distort the block time
        tz_shift = _tz_offset_hours(t["timezone"], arr) - _tz_offset_hours(o["timezone"], dep)
        block -= tz_shift * 60
    else:
        block = route.get("scheduled_block_min") or (30 + dist / 12.5)
    block = float(np.clip(block, 25, 600))
    f["scheduled_block_min"] = block
    implied = 30 + dist / 12.5
    f["schedule_padding_min"] = float(block - implied)
    f["padding_ratio"] = float((block - implied) / max(block, 1))

    f["scheduled_turnaround_min"] = float(
        req.get("scheduled_turnaround_min") or d["scheduled_turnaround_min"])

    # ---------------- aircraft -------------------------------------------
    tail = (req.get("tail_number") or "").strip().upper()
    ac = s["aircraft"].get(tail, {})
    f["aircraft_model"] = req.get("aircraft_model") or ac.get("aircraft_model") or d["aircraft_model"]
    f["seats"] = float(req.get("seats") or ac.get("seats") or d["seats"])
    f["aircraft_age_years"] = float(
        req.get("aircraft_age_years") if req.get("aircraft_age_years") is not None
        else ac.get("aircraft_age_years", d["aircraft_age_years"]))

    # ---------------- weather --------------------------------------------
    normals_o = s["weather_normals"].get(f"{origin}|{dep.month}", {})
    normals_d = s["weather_normals"].get(f"{dest}|{dep.month}", {})
    wo = _wx(req.get("origin_weather"), normals_o)
    wd = _wx(req.get("dest_weather"), normals_d)

    f["dep_wx_temperature_c"] = float(wo["temperature_c"])
    f["dep_wx_wind_speed_kt"] = float(wo["wind_speed_kt"])
    f["dep_wx_visibility_km"] = float(wo["visibility_km"])
    f["dep_wx_cloud_ceiling_ft"] = float(wo["cloud_ceiling_ft"])
    f["dep_dewpoint_spread"] = float(wo["temperature_c"] - wo["dewpoint_c"])
    f["dep_gust_factor"] = float(wo["wind_gust_kt"] / max(wo["wind_speed_kt"], 1))
    f["dep_temp_anomaly"] = float(
        wo["temperature_c"] - normals_o.get("dep_wx_temperature_c", wo["temperature_c"]))
    sev_o = wx_severity(wo["precip_mm"], wo["wind_gust_kt"], wo["visibility_km"],
                        wo["snow_mm"], wo["is_storm"], wo["cloud_ceiling_ft"])
    sev_d = wx_severity(wd["precip_mm"], wd["wind_gust_kt"], wd["visibility_km"],
                        wd["snow_mm"], wd["is_storm"], wd["cloud_ceiling_ft"])
    f["dep_wx_severity"] = sev_o
    f["route_wx_severity"] = sev_o + 0.5 * sev_d

    # ---------------- congestion -----------------------------------------
    dem = s["demand"].get(f"{origin}|{f['day_of_week']}|{dep.hour}", {})
    dep_count = dem.get("origin_hour_departures", d.get("origin_hour_departures", 4))
    f["origin_congestion_ratio"] = float(dep_count * 6.0 / max(o["hourly_capacity"], 1))

    co = s["carrier_origin"].get(f"{carrier}|{origin}", {})
    f["carrier_share_at_origin"] = float(
        co.get("carrier_share_at_origin", d["carrier_share_at_origin"]))

    # ---------------- historical -----------------------------------------
    car = s["carrier"].get(carrier, {})
    f["carrier_delay_rate_7d"] = float(car.get("carrier_delay_rate_7d", d["carrier_delay_rate_7d"]))
    f["carrier_volume_7d"] = float(car.get("carrier_volume_7d", d["carrier_volume_7d"]))
    orig = s["origin"].get(origin, {})
    f["origin_delay_rate_7d"] = float(orig.get("origin_delay_rate_7d", d["origin_delay_rate_7d"]))
    f["origin_mean_delay_min_7d"] = float(
        orig.get("origin_mean_delay_min_7d", d["origin_mean_delay_min_7d"]))
    f["route_delay_rate_30d"] = float(route.get("route_delay_rate_30d", d["route_delay_rate_30d"]))
    f["route_popularity_30d"] = float(route.get("route_popularity_30d", d["route_popularity_30d"]))
    f["carrier_origin_delay_rate_14d"] = float(
        co.get("carrier_origin_delay_rate_14d", d["carrier_origin_delay_rate_14d"]))
    f["carrier_origin_volume_14d"] = float(
        co.get("carrier_origin_volume_14d", d["carrier_origin_volume_14d"]))

    # ---------------- upstream propagation --------------------------------
    prev_arr = req.get("prev_leg_arr_delay_min")
    prev_dep = req.get("prev_leg_dep_delay_min")
    known = prev_arr is not None or prev_dep is not None
    upstream = max(float(prev_arr if prev_arr is not None else (prev_dep or 0.0)), 0.0)
    f["prev_leg_known"] = int(prev_arr is not None)
    f["upstream_delay_min"] = upstream
    ground = req.get("scheduled_ground_time_min")
    ground = float(ground) if ground is not None else (
        f["scheduled_turnaround_min"] if known else 999.0)
    f["slack_vs_upstream_min"] = float(ground - 30 - upstream)
    f["aircraft_cum_delay_today"] = float(req.get("aircraft_cum_delay_today") or 0.0)

    # ---------------- identifiers ----------------------------------------
    f["carrier_code"] = carrier
    f["origin"] = origin
    f["destination"] = dest

    # anything still missing falls back to the training median / mode
    for k, v in d.items():
        f.setdefault(k, v)
    return f


def _tz_offset_hours(tz: str, when: pd.Timestamp) -> float:
    from zoneinfo import ZoneInfo
    naive = when.to_pydatetime().replace(tzinfo=None)
    return ZoneInfo(tz).utcoffset(naive).total_seconds() / 3600


def to_frame(feats: dict | list[dict], features: list[str],
             categories: dict[str, list]) -> pd.DataFrame:
    """Order columns and restore the exact dtypes the model was fitted on."""
    rows = feats if isinstance(feats, list) else [feats]
    X = pd.DataFrame(rows)
    for c in features:
        if c not in X.columns:
            X[c] = np.nan
    X = X[features]
    for c in features:
        if c in categories:
            X[c] = pd.Categorical(X[c].astype("object"), categories=categories[c])
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X
