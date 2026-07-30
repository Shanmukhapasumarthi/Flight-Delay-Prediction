"""Stage 5 - Feature Engineering.

Six families of engineered features are built here. The organising principle
is *information availability*: every feature must be computable at the moment
the prediction is made (a few hours before scheduled departure).

  1. TIME        hour, part-of-day, weekend, season, holiday proximity, ...
  2. WEATHER     intensity bands, IFR/LIFR flags, temperature anomaly,
                 a composite severity score for origin and destination
  3. AIRPORT     scheduled demand vs. declared capacity (congestion), bank
                 pressure, carrier dominance, runway/elevation
  4. FLIGHT      schedule padding, leg depth in rotation, turnaround buffer,
                 aircraft age/seats/category, route popularity
  5. GEOSPATIAL  coordinates, bearing, direction of travel, region pair,
                 hub-to-hub, distance bands
  6. HISTORICAL  strictly-prior rolling delay rates (carrier 7d, route 30d,
                 airport 7d, carrier x airport 14d) and, most importantly,
                 UPSTREAM DELAY PROPAGATION from the aircraft's previous leg

LEAKAGE CONTROL
---------------
* All rolling aggregates use a time-based window with `closed="left"`, so the
  current day never contributes to its own feature.
* Previous-leg arrival delay is only used when the previous leg actually
  landed before the current flight's scheduled departure; otherwise the
  (always-known) previous departure delay is used and a flag is set.
* The chronological train/valid/test split means no future day can influence
  a past prediction.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import CFG, DELAY_THRESHOLD, INTERIM, PROCESSED, REPORTS

RW = CFG["features"]["rolling_windows"]


# --------------------------------------------------------------------------
# 1. TIME
# --------------------------------------------------------------------------
def add_time_features(f: pd.DataFrame) -> pd.DataFrame:
    local = pd.to_datetime(f["scheduled_departure_local"])
    f["dep_hour"] = local.dt.hour.astype("int8")
    f["dep_minute"] = local.dt.minute.astype("int8")
    f["dep_time_decimal"] = (local.dt.hour + local.dt.minute / 60).astype("float32")
    f["day_of_week"] = local.dt.dayofweek.astype("int8")          # 0 = Monday
    f["day_of_month"] = local.dt.day.astype("int8")
    f["month"] = local.dt.month.astype("int8")
    f["quarter"] = local.dt.quarter.astype("int8")
    f["day_of_year"] = local.dt.dayofyear.astype("int16")
    f["week_of_year"] = local.dt.isocalendar().week.astype("int16")
    f["is_weekend"] = (f["day_of_week"] >= 5).astype("int8")
    f["is_monday"] = (f["day_of_week"] == 0).astype("int8")
    f["is_friday"] = (f["day_of_week"] == 4).astype("int8")

    # operational time-of-day bands
    f["is_red_eye"] = ((f["dep_hour"] >= 22) | (f["dep_hour"] <= 4)).astype("int8")
    f["is_morning_bank"] = f["dep_hour"].between(6, 9).astype("int8")
    f["is_evening_bank"] = f["dep_hour"].between(16, 20).astype("int8")
    f["is_peak_hour"] = f["dep_hour"].isin([7, 8, 17, 18, 19]).astype("int8")
    f["time_block"] = pd.cut(f["dep_hour"], [-1, 5, 9, 12, 16, 20, 24],
                             labels=["night", "early_morning", "late_morning",
                                     "afternoon", "evening", "late_evening"])

    # cyclical encodings so that 23:00 and 00:00 are close together
    f["hour_sin"] = np.sin(2 * np.pi * f["dep_time_decimal"] / 24).astype("float32")
    f["hour_cos"] = np.cos(2 * np.pi * f["dep_time_decimal"] / 24).astype("float32")
    f["doy_sin"] = np.sin(2 * np.pi * f["day_of_year"] / 365.25).astype("float32")
    f["doy_cos"] = np.cos(2 * np.pi * f["day_of_year"] / 365.25).astype("float32")

    f["season"] = f["month"].map({12: "winter", 1: "winter", 2: "winter",
                                  3: "spring", 4: "spring", 5: "spring",
                                  6: "summer", 7: "summer", 8: "summer",
                                  9: "fall", 10: "fall", 11: "fall"})
    f["is_summer_peak"] = f["month"].isin([6, 7, 8]).astype("int8")
    f["is_winter_ops"] = f["month"].isin([12, 1, 2]).astype("int8")

    d = f["days_from_major_holiday"]
    f["days_to_holiday"] = (-d).clip(lower=0).where(d <= 0, 0).astype("float32")
    f["days_after_holiday"] = d.clip(lower=0).astype("float32")
    f["in_holiday_window"] = (d.abs() <= 3).astype("int8")
    f["holiday_proximity"] = np.clip(1 - d.abs() / 4.0, 0, 1).astype("float32")
    return f


# --------------------------------------------------------------------------
# 2. WEATHER
# --------------------------------------------------------------------------
def _weather_block(f: pd.DataFrame, p: str) -> pd.DataFrame:
    """Engineer weather features for prefix p ('dep_wx_' or 'arr_wx_')."""
    s = p.replace("_wx_", "")          # 'dep' / 'arr'
    precip = f[p + "precip_mm"].fillna(0)
    snow = f[p + "snow_mm"].fillna(0)
    gust = f[p + "wind_gust_kt"].fillna(f[p + "wind_speed_kt"])
    wind = f[p + "wind_speed_kt"].fillna(0)
    vis = f[p + "visibility_km"].fillna(16)
    ceil = f[p + "cloud_ceiling_ft"].fillna(20000)
    temp = f[p + "temperature_c"]
    dew = f[p + "dewpoint_c"]

    f[f"{s}_rain_intensity"] = pd.cut(
        precip, [-0.01, 0.1, 2.5, 7.6, 1000],
        labels=["none", "light", "moderate", "heavy"])
    f[f"{s}_wind_category"] = pd.cut(
        gust, [-0.01, 12, 22, 34, 1000],
        labels=["calm", "breezy", "strong", "gale"])
    f[f"{s}_visibility_level"] = pd.cut(
        vis, [-0.01, 1.6, 5, 10, 100],
        labels=["very_poor", "poor", "moderate", "good"])

    f[f"{s}_is_storm"] = ((f[p + "condition"] == "Thunderstorm") |
                          ((gust > 35) & (precip > 3))).astype("int8")
    f[f"{s}_is_snow"] = (snow > 0.5).astype("int8")
    f[f"{s}_is_freezing"] = (temp < 0).astype("int8")
    # instrument / low-instrument flight rules -> reduced arrival rates
    f[f"{s}_ifr"] = ((ceil < 3000) | (vis < 8)).astype("int8")
    f[f"{s}_lifr"] = ((ceil < 500) | (vis < 1.6)).astype("int8")
    f[f"{s}_dewpoint_spread"] = (temp - dew).astype("float32")   # fog proxy
    f[f"{s}_fog_risk"] = ((temp - dew < 2) & (vis < 5)).astype("int8")
    f[f"{s}_gust_factor"] = (gust / wind.clip(lower=1)).astype("float32")
    f[f"{s}_heat_stress"] = (temp - 32).clip(lower=0).astype("float32")

    # composite operational severity (engineered, not observed)
    f[f"{s}_wx_severity"] = (
        0.85 * np.clip(precip / 6.0, 0, 1.7)
        + 1.25 * np.clip((gust - 24) / 26.0, 0, 1.6)
        + 1.05 * np.clip((5.0 - vis) / 5.0, 0, 1.0)
        + 1.55 * np.clip(snow / 18.0, 0, 1.7)
        + 1.15 * f[f"{s}_is_storm"]
        + 0.35 * (ceil < 800).astype(float)
    ).astype("float32")
    return f


def add_weather_features(f: pd.DataFrame) -> pd.DataFrame:
    f = _weather_block(f, "dep_wx_")
    f = _weather_block(f, "arr_wx_")
    f["route_wx_severity"] = (f["dep_wx_severity"] + 0.5 * f["arr_wx_severity"]).astype("float32")
    f["both_ends_bad_wx"] = ((f["dep_ifr"] == 1) & (f["arr_ifr"] == 1)).astype("int8")

    # temperature anomaly vs. the airport's own monthly normal
    norm = (f.groupby(["origin", "month"], observed=True)["dep_wx_temperature_c"]
              .transform("mean"))
    f["dep_temp_anomaly"] = (f["dep_wx_temperature_c"] - norm).astype("float32")
    return f


# --------------------------------------------------------------------------
# 3. AIRPORT / CONGESTION  (from the published schedule -> known in advance)
# --------------------------------------------------------------------------
def add_airport_features(f: pd.DataFrame) -> pd.DataFrame:
    dep_h = f["scheduled_departure_utc"].dt.floor("h")
    arr_h = f["scheduled_arrival_utc"].dt.floor("h")
    f["_dep_h"] = dep_h
    f["_arr_h"] = arr_h

    dep_demand = f.groupby(["origin", "_dep_h"]).size().rename("origin_hour_departures")
    arr_demand = f.groupby(["destination", "_arr_h"]).size().rename("dest_hour_arrivals")
    f = f.merge(dep_demand, left_on=["origin", "_dep_h"], right_index=True, how="left")
    f = f.merge(arr_demand, left_on=["destination", "_arr_h"], right_index=True, how="left")

    # the simulated network is a sample of real traffic; the ratio is what
    # matters, so demand is expressed relative to declared hourly capacity
    f["origin_congestion_ratio"] = (f["origin_hour_departures"] * 6.0 /
                                    f["origin_hourly_capacity"]).astype("float32")
    f["dest_congestion_ratio"] = (f["dest_hour_arrivals"] * 6.0 /
                                  f["dest_hourly_capacity"]).astype("float32")
    f["origin_over_capacity"] = (f["origin_congestion_ratio"] > 1).astype("int8")

    # bank pressure: departures in the 3-hour window centred on this hour
    hourly = dep_demand.reset_index().rename(columns={"_dep_h": "h"})
    hourly = hourly.sort_values(["origin", "h"])
    hourly["bank_pressure_3h"] = (hourly.groupby("origin")["origin_hour_departures"]
                                  .transform(lambda s: s.rolling(3, center=True,
                                                                 min_periods=1).sum()))
    f = f.merge(hourly[["origin", "h", "bank_pressure_3h"]],
                left_on=["origin", "_dep_h"], right_on=["origin", "h"], how="left")
    f = f.drop(columns=["h"])

    # daily airport volume and carrier dominance at the origin
    daily = f.groupby(["origin", "local_departure_date"]).size().rename("origin_daily_departures")
    f = f.merge(daily, left_on=["origin", "local_departure_date"], right_index=True, how="left")
    car_share = (f.groupby(["origin", "carrier_code"]).size() /
                 f.groupby("origin").size()).rename("carrier_share_at_origin")
    f = f.merge(car_share, left_on=["origin", "carrier_code"], right_index=True, how="left")

    f["runways_per_departure"] = (f["origin_num_runways"] /
                                  f["origin_hour_departures"].clip(lower=1)).astype("float32")
    f["origin_is_slot_constrained"] = f["origin"].isin(["JFK", "LGA", "DCA", "EWR"]).astype("int8")
    return f


# --------------------------------------------------------------------------
# 4. FLIGHT
# --------------------------------------------------------------------------
def add_flight_features(f: pd.DataFrame) -> pd.DataFrame:
    f["distance_band"] = pd.cut(f["distance_km"], [0, 500, 1200, 2500, 9000],
                                labels=["short", "medium", "long", "transcon"])
    # schedule padding: block time minus the physically implied flying time
    implied = 30 + f["distance_km"] / 12.5
    f["schedule_padding_min"] = (f["scheduled_block_min"] - implied).astype("float32")
    f["padding_ratio"] = (f["schedule_padding_min"] /
                          f["scheduled_block_min"].clip(lower=1)).astype("float32")
    f["speed_kmh_implied"] = (f["distance_km"] /
                              (f["scheduled_block_min"] / 60).clip(lower=0.1)).astype("float32")
    f["turnaround_buffer_min"] = (f["scheduled_turnaround_min"] - 30).astype("float32")
    f["tight_turnaround"] = (f["scheduled_turnaround_min"] < 45).astype("int8")
    f["leg_depth"] = f["leg_number"].astype("int8")
    f["is_first_leg_of_day"] = (f["leg_number"] == 0).astype("int8")
    f["aircraft_age_years"] = f["aircraft_age_years"].astype("float32")
    f["is_old_aircraft"] = (f["aircraft_age_years"] > 18).astype("int8")
    f["seats"] = f["seats"].astype("float32")
    f["is_regional_jet"] = (f["category"] == "regional_jet").astype("int8")
    f["route"] = f["origin"] + "-" + f["destination"]
    return f


# --------------------------------------------------------------------------
# 5. GEOSPATIAL
# --------------------------------------------------------------------------
def add_geo_features(f: pd.DataFrame) -> pd.DataFrame:
    lat1, lon1 = np.radians(f["origin_latitude"]), np.radians(f["origin_longitude"])
    lat2, lon2 = np.radians(f["dest_latitude"]), np.radians(f["dest_longitude"])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360
    f["route_bearing_deg"] = bearing.astype("float32")
    f["bearing_sin"] = np.sin(np.radians(bearing)).astype("float32")
    f["bearing_cos"] = np.cos(np.radians(bearing)).astype("float32")
    f["is_eastbound"] = ((bearing > 45) & (bearing < 135)).astype("int8")
    f["is_westbound"] = ((bearing > 225) & (bearing < 315)).astype("int8")
    f["lat_delta"] = (f["dest_latitude"] - f["origin_latitude"]).astype("float32")
    f["lon_delta"] = (f["dest_longitude"] - f["origin_longitude"]).astype("float32")
    f["same_region"] = (f["origin_region"] == f["dest_region"]).astype("int8")
    f["hub_to_hub"] = (f["origin_is_hub"] * f["dest_is_hub"]).astype("int8")
    f["hub_pair_type"] = np.select(
        [(f.origin_is_hub == 1) & (f.dest_is_hub == 1),
         (f.origin_is_hub == 1) & (f.dest_is_hub == 0),
         (f.origin_is_hub == 0) & (f.dest_is_hub == 1)],
        ["hub_hub", "hub_spoke", "spoke_hub"], default="spoke_spoke")
    f["elevation_gain_ft"] = (f["dest_elevation_ft"] - f["origin_elevation_ft"]).astype("float32")
    f["high_altitude_origin"] = (f["origin_elevation_ft"] > 4000).astype("int8")
    return f


# --------------------------------------------------------------------------
# 6. HISTORICAL  (strictly prior information only)
# --------------------------------------------------------------------------
def _rolling_prior(daily: pd.DataFrame, key: list[str], window: str,
                   out_rate: str, out_vol: str) -> pd.DataFrame:
    """Time-based rolling mean over *strictly previous* days."""
    d = daily.sort_values(key + ["date"]).set_index("date")
    grp = d.groupby(key, observed=True)[["n_delayed", "n_flights"]]
    roll = grp.rolling(window, closed="left").sum()
    roll[out_rate] = (roll["n_delayed"] / roll["n_flights"]).astype("float32")
    roll[out_vol] = roll["n_flights"].astype("float32")
    return roll[[out_rate, out_vol]].reset_index()


def add_historical_features(f: pd.DataFrame) -> pd.DataFrame:
    f["date"] = pd.to_datetime(f["local_departure_date"])
    base = f.loc[f["cancelled"] == 0, ["date", "carrier_code", "origin", "route",
                                       "is_delayed", "departure_delay_min"]]

    specs = [
        (["carrier_code"], f"{RW['carrier_days']}D", "carrier_delay_rate_7d", "carrier_volume_7d"),
        (["route"], f"{RW['route_days']}D", "route_delay_rate_30d", "route_volume_30d"),
        (["origin"], f"{RW['airport_days']}D", "origin_delay_rate_7d", "origin_volume_7d"),
        (["carrier_code", "origin"], "14D", "carrier_origin_delay_rate_14d", "carrier_origin_volume_14d"),
    ]
    for key, win, rate_col, vol_col in specs:
        daily = (base.groupby(key + ["date"], observed=True)
                     .agg(n_delayed=("is_delayed", "sum"), n_flights=("is_delayed", "size"))
                     .reset_index())
        roll = _rolling_prior(daily, key, win, rate_col, vol_col)
        f = f.merge(roll, on=key + ["date"], how="left")

    # mean delay MINUTES at the origin over the previous week
    daily_min = (base.groupby(["origin", "date"], observed=True)["departure_delay_min"]
                     .mean().reset_index(name="mean_delay"))
    d = daily_min.sort_values(["origin", "date"]).set_index("date")
    roll = (d.groupby("origin")["mean_delay"].rolling("7D", closed="left").mean()
             .rename("origin_mean_delay_min_7d").reset_index())
    f = f.merge(roll, on=["origin", "date"], how="left")

    f["route_popularity_30d"] = f["route_volume_30d"].fillna(0).astype("float32")
    f["carrier_vs_system_delay"] = (f["carrier_delay_rate_7d"] -
                                    f["carrier_delay_rate_7d"].mean()).astype("float32")
    return f


def add_propagation_features(f: pd.DataFrame) -> pd.DataFrame:
    """Upstream delay propagation from the same airframe's previous leg."""
    f = f.sort_values(["tail_number", "scheduled_departure_utc"]).copy()
    g = f.groupby("tail_number", sort=False)

    prev_dep_delay = g["departure_delay_min"].shift(1)
    prev_arr_delay = g["arrival_delay_min"].shift(1)
    prev_sched_arr = g["scheduled_arrival_utc"].shift(1)
    prev_dest = g["destination"].shift(1)
    prev_cancelled = g["cancelled"].shift(1)

    actual_arr = prev_sched_arr + pd.to_timedelta(prev_arr_delay.fillna(0), unit="m")
    # is the previous leg on the ground before we are scheduled out?
    known = (actual_arr <= f["scheduled_departure_utc"]) & prev_arr_delay.notna()
    same_airframe_chain = (prev_dest == f["origin"]) & (prev_cancelled.fillna(1) == 0)

    f["prev_leg_arr_delay_min"] = np.where(known & same_airframe_chain,
                                           prev_arr_delay, np.nan)
    f["prev_leg_dep_delay_min"] = np.where(same_airframe_chain, prev_dep_delay, np.nan)
    f["prev_leg_known"] = (known & same_airframe_chain).astype("int8")
    f["upstream_delay_min"] = (f["prev_leg_arr_delay_min"]
                               .fillna(f["prev_leg_dep_delay_min"])
                               .fillna(0).clip(lower=0).astype("float32"))
    f["upstream_delayed_15"] = (f["upstream_delay_min"] >= 15).astype("int8")

    # ground time actually available between the two legs
    gt = (f["scheduled_departure_utc"] - prev_sched_arr).dt.total_seconds() / 60
    f["scheduled_ground_time_min"] = gt.where(same_airframe_chain).astype("float32")
    f["slack_vs_upstream_min"] = (f["scheduled_ground_time_min"].fillna(999) - 30
                                  - f["upstream_delay_min"]).astype("float32")
    f["at_risk_of_propagation"] = ((f["slack_vs_upstream_min"] < 0) &
                                   (f["prev_leg_known"] == 1)).astype("int8")

    # how much delay has this airframe already accumulated today
    f["_d"] = f["local_departure_date"]
    f["aircraft_cum_delay_today"] = (
        f.groupby(["tail_number", "_d"], sort=False)["departure_delay_min"]
         .transform(lambda s: s.shift(1).cumsum())).fillna(0).astype("float32")
    f = f.drop(columns=["_d"])
    return f.sort_values("scheduled_departure_utc").reset_index(drop=True)


# --------------------------------------------------------------------------
FEATURE_GROUPS = {
    "time": ["dep_hour", "dep_minute", "dep_time_decimal", "day_of_week",
             "day_of_month", "month", "quarter", "day_of_year", "week_of_year",
             "is_weekend", "is_monday", "is_friday", "is_red_eye",
             "is_morning_bank", "is_evening_bank", "is_peak_hour", "hour_sin",
             "hour_cos", "doy_sin", "doy_cos", "is_summer_peak",
             "is_winter_ops", "days_to_holiday", "days_after_holiday",
             "in_holiday_window", "holiday_proximity", "is_holiday",
             "is_major_travel_holiday", "time_block", "season"],
    "weather": ["dep_wx_temperature_c", "dep_wx_wind_speed_kt", "dep_wx_wind_gust_kt",
                "dep_wx_precip_mm", "dep_wx_snow_mm", "dep_wx_visibility_km",
                "dep_wx_cloud_ceiling_ft", "dep_rain_intensity", "dep_wind_category",
                "dep_visibility_level", "dep_is_storm", "dep_is_snow",
                "dep_is_freezing", "dep_ifr", "dep_lifr", "dep_dewpoint_spread",
                "dep_fog_risk", "dep_gust_factor", "dep_heat_stress",
                "dep_wx_severity", "arr_wx_precip_mm", "arr_wx_wind_gust_kt",
                "arr_wx_visibility_km", "arr_is_storm", "arr_ifr",
                "arr_wx_severity", "route_wx_severity", "both_ends_bad_wx",
                "dep_temp_anomaly", "dep_wx_condition", "arr_wx_condition"],
    "airport": ["origin_hour_departures", "dest_hour_arrivals",
                "origin_congestion_ratio", "dest_congestion_ratio",
                "origin_over_capacity", "bank_pressure_3h",
                "origin_daily_departures", "carrier_share_at_origin",
                "origin_num_runways", "origin_hourly_capacity",
                "runways_per_departure", "origin_is_slot_constrained",
                "origin_elevation_ft", "origin_is_hub", "dest_is_hub"],
    "flight": ["distance_km", "distance_band", "scheduled_block_min",
               "schedule_padding_min", "padding_ratio", "speed_kmh_implied",
               "scheduled_turnaround_min", "turnaround_buffer_min",
               "tight_turnaround", "leg_depth", "is_first_leg_of_day",
               "aircraft_age_years", "is_old_aircraft", "seats",
               "is_regional_jet", "carrier_code", "carrier_type",
               "aircraft_model", "category"],
    "geospatial": ["origin_latitude", "origin_longitude", "dest_latitude",
                   "dest_longitude", "route_bearing_deg", "bearing_sin",
                   "bearing_cos", "is_eastbound", "is_westbound", "lat_delta",
                   "lon_delta", "same_region", "hub_to_hub", "hub_pair_type",
                   "elevation_gain_ft", "high_altitude_origin",
                   "origin_region", "dest_region", "origin", "destination"],
    "historical": ["carrier_delay_rate_7d", "carrier_volume_7d",
                   "route_delay_rate_30d", "route_volume_30d",
                   "origin_delay_rate_7d", "origin_volume_7d",
                   "carrier_origin_delay_rate_14d", "carrier_origin_volume_14d",
                   "origin_mean_delay_min_7d", "route_popularity_30d",
                   "carrier_vs_system_delay", "prev_leg_arr_delay_min",
                   "prev_leg_dep_delay_min", "prev_leg_known",
                   "upstream_delay_min", "upstream_delayed_15",
                   "scheduled_ground_time_min", "slack_vs_upstream_min",
                   "at_risk_of_propagation", "aircraft_cum_delay_today"],
}

ID_COLS = ["flight_id", "carrier_code", "flight_number", "tail_number", "origin",
           "destination", "route", "scheduled_departure_utc",
           "scheduled_departure_local", "local_departure_date", "date",
           "departure_delay_min", "arrival_delay_min", "is_delayed",
           "cancelled", "diverted", "carrier_name", "holiday_name",
           "days_from_major_holiday"]


def main() -> None:
    print("[features] loading integrated data")
    f = pd.read_parquet(INTERIM / "integrated.parquet")
    f["is_delayed"] = (f["departure_delay_min"] >= DELAY_THRESHOLD).astype("int8")

    for name, fn in [("time", add_time_features),
                     ("weather", add_weather_features),
                     ("airport", add_airport_features),
                     ("flight", add_flight_features),
                     ("geospatial", add_geo_features),
                     ("historical", add_historical_features),
                     ("propagation", add_propagation_features)]:
        f = fn(f)
        print(f"[features] {name:<12} -> {f.shape[1]} columns")

    f = f.drop(columns=[c for c in ("_dep_h", "_arr_h") if c in f.columns])

    feature_cols = sorted({c for cols in FEATURE_GROUPS.values() for c in cols})
    missing = [c for c in feature_cols if c not in f.columns]
    if missing:
        raise RuntimeError(f"declared but not built: {missing}")

    keep = [c for c in dict.fromkeys(ID_COLS + feature_cols) if c in f.columns]
    out = f[keep]

    # modelling frame excludes cancellations (no departure delay exists)
    model_df = out[out["cancelled"] == 0].reset_index(drop=True)
    model_df.to_parquet(PROCESSED / "features.parquet", index=False)
    out.to_parquet(PROCESSED / "features_all.parquet", index=False)

    manifest = {
        "n_rows": int(len(model_df)),
        "n_features": len(feature_cols),
        "target": "is_delayed",
        "delay_threshold_min": DELAY_THRESHOLD,
        "positive_rate": float(model_df["is_delayed"].mean()),
        "groups": {k: sorted(v) for k, v in FEATURE_GROUPS.items()},
        "group_sizes": {k: len(v) for k, v in FEATURE_GROUPS.items()},
    }
    (REPORTS / "feature_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                   encoding="utf-8")
    print(f"[features] {len(model_df):,} modelling rows, {len(feature_cols)} features")
    print(f"[features] positive rate {model_df['is_delayed'].mean():.2%}")
    for k, v in manifest["group_sizes"].items():
        print(f"             {k:<12} {v:>3}")


if __name__ == "__main__":
    main()
