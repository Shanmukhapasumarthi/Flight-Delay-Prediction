"""Online feature store.

Training features are computed over the whole history at once. At serving
time we have a single flight and no dataframe of the past, so the rolling
aggregates must be looked up. This module freezes the most recent state of
every historical aggregate into one artifact that the API loads at start-up.

This is the standard defence against train/serve skew: the SAME derived
quantity must be available offline and online, computed the same way.

Artifact: models/feature_store.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import INTERIM, MODELS_DIR, PROCESSED
from src.models.dataset import CATEGORICAL, ALL_FEATURES, selected_features


def _clean(o):
    """JSON-safe conversion."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if pd.isna(o) if np.isscalar(o) else False:
        return None
    return o


def build(df: pd.DataFrame) -> dict:
    latest = df.sort_values("date").groupby("carrier_code").tail(1)
    store: dict = {"as_of": str(df["date"].max().date())}

    # ---- reference -------------------------------------------------------
    airports = pd.read_parquet(INTERIM / "airports.parquet").set_index("airport_code")
    store["airports"] = _clean(airports.to_dict("index"))

    aircraft = pd.read_parquet(INTERIM / "aircraft.parquet").set_index("tail_number")
    store["aircraft"] = _clean(aircraft[["aircraft_model", "seats", "category",
                                         "age_years_2024", "carrier_code"]]
                               .rename(columns={"age_years_2024": "aircraft_age_years"})
                               .to_dict("index"))

    airlines = pd.read_parquet(INTERIM / "airlines.parquet").set_index("carrier_code")
    store["carrier_type"] = _clean(airlines["carrier_type"].to_dict())

    # ---- latest rolling aggregates --------------------------------------
    store["carrier"] = _clean(
        latest.set_index("carrier_code")[["carrier_delay_rate_7d", "carrier_volume_7d"]]
        .to_dict("index"))

    org = df.sort_values("date").groupby("origin").tail(1).set_index("origin")
    store["origin"] = _clean(org[["origin_delay_rate_7d", "origin_volume_7d",
                                  "origin_mean_delay_min_7d"]].to_dict("index"))

    rt = df.sort_values("date").groupby("route").tail(1).set_index("route")
    store["route"] = _clean(rt[["route_delay_rate_30d", "route_popularity_30d",
                                "distance_km", "scheduled_block_min"]].to_dict("index"))

    co = df.sort_values("date").groupby(["carrier_code", "origin"]).tail(1)
    co["key"] = co["carrier_code"] + "|" + co["origin"]
    store["carrier_origin"] = _clean(
        co.set_index("key")[["carrier_origin_delay_rate_14d",
                             "carrier_origin_volume_14d",
                             "carrier_share_at_origin"]].to_dict("index"))

    # ---- typical schedule pressure by airport / weekday / hour ----------
    dem = (df.groupby(["origin", "day_of_week", "dep_hour"])
             .agg(origin_hour_departures=("flight_id", "size"),
                  bank_pressure_3h=("bank_pressure_3h", "mean"),
                  origin_daily_departures=("origin_daily_departures", "mean"),
                  n_days=("date", "nunique")).reset_index())
    dem["origin_hour_departures"] = dem["origin_hour_departures"] / dem["n_days"].clip(lower=1)
    dem["key"] = (dem["origin"] + "|" + dem["day_of_week"].astype(str)
                  + "|" + dem["dep_hour"].astype(str))
    store["demand"] = _clean(dem.set_index("key")[
        ["origin_hour_departures", "bank_pressure_3h", "origin_daily_departures"]
    ].round(3).to_dict("index"))

    arr = (df.groupby(["destination", "day_of_week"])["dest_hour_arrivals"]
             .mean().reset_index())
    arr["key"] = arr["destination"] + "|" + arr["day_of_week"].astype(str)
    store["dest_arrivals"] = _clean(arr.set_index("key")["dest_hour_arrivals"]
                                    .round(2).to_dict())

    # ---- defaults for anything the caller omits --------------------------
    feats = selected_features() or ALL_FEATURES
    defaults = {}
    for f in feats:
        if f not in df.columns:
            continue
        s = df[f]
        if f in CATEGORICAL or str(s.dtype) in ("object", "category", "string"):
            m = s.mode()
            defaults[f] = None if m.empty else str(m.iloc[0])
        else:
            defaults[f] = float(pd.to_numeric(s, errors="coerce").median())
    store["defaults"] = _clean(defaults)

    # climatological weather medians per airport & month (for missing forecasts)
    wx = (df.groupby(["origin", "month"])[
        ["dep_wx_temperature_c", "dep_wx_wind_speed_kt", "dep_wx_wind_gust_kt",
         "dep_wx_precip_mm", "dep_wx_visibility_km", "dep_wx_cloud_ceiling_ft"]]
        .median().round(2).reset_index())
    wx["key"] = wx["origin"] + "|" + wx["month"].astype(str)
    store["weather_normals"] = _clean(
        wx.set_index("key").drop(columns=["origin", "month"]).to_dict("index"))

    return store


def main() -> None:
    df = pd.read_parquet(PROCESSED / "features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    store = build(df)
    out = MODELS_DIR / "feature_store.json"
    out.write_text(json.dumps(store, indent=1), encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"[store] as_of {store['as_of']}  |  {size_kb:.0f} KB  ->  {out}")
    for k in ("airports", "aircraft", "carrier", "origin", "route",
              "carrier_origin", "demand", "weather_normals"):
        print(f"          {k:<16} {len(store[k]):>6,} entries")


if __name__ == "__main__":
    main()
