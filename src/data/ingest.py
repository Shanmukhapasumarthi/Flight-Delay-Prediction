"""Stage 2/3 - Data Integration.

Joins the six cleaned tables into one analysis-ready flight-level frame:

    flights
      + airports          (origin and destination geography, capacity)
      + airlines          (carrier reference)
      + aircraft          (fleet registry, age, seats)
      + weather_hourly    (origin @ departure hour, destination @ arrival hour)
      + holiday_calendar  (local departure date)

Note on the weather join: we attach the OBSERVATION at the scheduled hour.
In production this must be swapped for the TAF/forecast valid at that hour,
otherwise the model is trained on information it will not have at inference
time. The join key and column names are identical, so the swap is a one-line
change in `attach_weather`.
"""
from __future__ import annotations

import pandas as pd

from src.config import INTERIM

WX_COLS = ["temperature_c", "dewpoint_c", "wind_speed_kt", "wind_gust_kt",
           "precip_mm", "snow_mm", "visibility_km", "cloud_ceiling_ft",
           "condition", "weather_imputed"]

AIRPORT_COLS = ["latitude", "longitude", "timezone", "num_runways",
                "elevation_ft", "is_hub", "hourly_capacity", "region",
                "city", "state", "climate_zone"]


def attach_airports(f: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    a = airports.set_index("airport_code")[AIRPORT_COLS]
    f = f.merge(a.add_prefix("origin_"), left_on="origin", right_index=True, how="left")
    f = f.merge(a.add_prefix("dest_"), left_on="destination", right_index=True, how="left")
    return f


def attach_airlines(f: pd.DataFrame, airlines: pd.DataFrame) -> pd.DataFrame:
    a = airlines.set_index("carrier_code")[["carrier_name", "carrier_type"]]
    return f.merge(a, left_on="carrier_code", right_index=True, how="left")


def attach_aircraft(f: pd.DataFrame, aircraft: pd.DataFrame) -> pd.DataFrame:
    a = aircraft.set_index("tail_number")[
        ["aircraft_model", "manufacturer", "seats", "category", "age_years_2024"]]
    a = a.rename(columns={"age_years_2024": "aircraft_age_years"})
    return f.merge(a, left_on="tail_number", right_index=True, how="left")


def attach_weather(f: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.set_index(["airport_code", "obs_time_utc"])[WX_COLS]

    f["dep_hour_utc"] = f["scheduled_departure_utc"].dt.floor("h")
    f["arr_hour_utc"] = f["scheduled_arrival_utc"].dt.floor("h")

    f = f.merge(w.add_prefix("dep_wx_"), how="left",
                left_on=["origin", "dep_hour_utc"], right_index=True)
    f = f.merge(w.add_prefix("arr_wx_"), how="left",
                left_on=["destination", "arr_hour_utc"], right_index=True)
    return f


def attach_holidays(f: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    h = holidays.copy()
    h["holiday_date"] = pd.to_datetime(h["holiday_date"]).dt.date
    hol_dates = set(h["holiday_date"])
    major = set(h.loc[h.is_major_travel == 1, "holiday_date"])
    name_map = dict(zip(h["holiday_date"], h["holiday_name"]))

    local_date = f["local_departure_date"]
    f["is_holiday"] = local_date.isin(hol_dates).astype("int8")
    f["is_major_travel_holiday"] = local_date.isin(major).astype("int8")
    f["holiday_name"] = local_date.map(name_map).fillna("None")

    # signed distance in days to the nearest major travel holiday
    md = pd.Series(sorted(major))
    ld = pd.to_datetime(pd.Series(local_date.unique()))
    md_ts = pd.to_datetime(md)
    diffs = (ld.to_numpy()[:, None] - md_ts.to_numpy()[None, :]) / pd.Timedelta("1D")
    nearest = diffs[range(len(ld)), abs(diffs).argmin(axis=1)]
    lookup = dict(zip(ld.dt.date, nearest))
    f["days_from_major_holiday"] = local_date.map(lookup).astype("float32")
    return f


def main() -> None:
    print("[ingest] loading cleaned tables")
    flights = pd.read_parquet(INTERIM / "flights.parquet")
    airports = pd.read_parquet(INTERIM / "airports.parquet")
    airlines = pd.read_parquet(INTERIM / "airlines.parquet")
    aircraft = pd.read_parquet(INTERIM / "aircraft.parquet")
    weather = pd.read_parquet(INTERIM / "weather.parquet")
    holidays = pd.read_parquet(INTERIM / "holidays.parquet")

    f = flights
    f["local_departure_date"] = pd.to_datetime(
        f["scheduled_departure_local"]).dt.date

    f = attach_airports(f, airports)
    print(f"[ingest] + airports        -> {f.shape[1]} cols")
    f = attach_airlines(f, airlines)
    print(f"[ingest] + airlines        -> {f.shape[1]} cols")
    f = attach_aircraft(f, aircraft)
    print(f"[ingest] + aircraft        -> {f.shape[1]} cols")
    f = attach_weather(f, weather)
    print(f"[ingest] + weather (o & d) -> {f.shape[1]} cols")
    f = attach_holidays(f, holidays)
    print(f"[ingest] + holidays        -> {f.shape[1]} cols")

    miss = f[["dep_wx_precip_mm", "arr_wx_precip_mm", "aircraft_age_years"]].isna().mean()
    print("[ingest] post-join missing rates:")
    for k, v in miss.items():
        print(f"           {k:<24} {v:.3%}")

    out = INTERIM / "integrated.parquet"
    f.to_parquet(out, index=False)
    print(f"[ingest] {len(f):,} rows x {f.shape[1]} cols -> {out}")


if __name__ == "__main__":
    main()
