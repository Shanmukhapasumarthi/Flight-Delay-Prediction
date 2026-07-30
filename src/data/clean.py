"""Stage 3 - Data Cleaning & Validation.

Handles every defect that exists in the raw feeds:

  flights   duplicate rows, dirty/invalid airport codes, arrival-before-
            departure timestamps, sentinel distances, absurd delay values,
            missing tail numbers, LOCAL naive timestamps that must be
            aligned to a single UTC axis
  weather   duplicated observations, missing hours, -999 sentinels,
            gaps that need bounded interpolation

Every action is counted and written to reports/data_quality_report.md so the
cleaning is auditable rather than silent.
"""
from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np
import pandas as pd

from src.config import INTERIM, RAW, REPORTS

SENTINELS = (-999, -999.0, -9999, 99999)
MAX_PLAUSIBLE_DELAY_MIN = 1440       # 24h; beyond this it is a data error
MIN_PLAUSIBLE_DELAY_MIN = -120


class QualityLog(OrderedDict):
    def add(self, step: str, n: int, detail: str = "") -> None:
        self[step] = {"rows_affected": int(n), "detail": detail}
        print(f"  [clean] {step:<38} {n:>8,}  {detail}")


# --------------------------------------------------------------------------
def clean_airports(df: pd.DataFrame, log: QualityLog) -> pd.DataFrame:
    df = df.copy()
    df["airport_code"] = df["airport_code"].str.strip().str.upper()
    before = len(df)
    df = df.drop_duplicates("airport_code")
    log.add("airports: duplicate codes removed", before - len(df))
    assert df["airport_code"].str.fullmatch(r"[A-Z]{3}").all(), "invalid IATA code"
    return df


def clean_aircraft(df: pd.DataFrame, log: QualityLog) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates("tail_number")
    log.add("aircraft: duplicate tail numbers removed", before - len(df))
    df["age_years_2024"] = df["age_years_2024"].clip(0, 45)
    return df


def clean_weather(df: pd.DataFrame, airports: pd.DataFrame,
                  log: QualityLog) -> pd.DataFrame:
    df = df.copy()
    df["airport_code"] = df["airport_code"].str.strip().str.upper()
    df["obs_time_utc"] = pd.to_datetime(df["obs_time_utc"], utc=True, format="mixed")

    before = len(df)
    df = df.drop_duplicates(["airport_code", "obs_time_utc"], keep="first")
    log.add("weather: duplicate observations removed", before - len(df))

    valid = set(airports["airport_code"])
    before = len(df)
    df = df[df["airport_code"].isin(valid)]
    log.add("weather: rows for unknown airports dropped", before - len(df))

    num_cols = ["temperature_c", "dewpoint_c", "wind_speed_kt", "wind_gust_kt",
                "precip_mm", "snow_mm", "visibility_km", "cloud_ceiling_ft"]
    n_sent = 0
    for c in num_cols:
        mask = df[c].isin(SENTINELS)
        n_sent += int(mask.sum())
        df.loc[mask, c] = np.nan
    log.add("weather: sentinel values -> NaN", n_sent, "(-999 / 99999)")

    # physical plausibility bounds
    bounds = {"temperature_c": (-60, 60), "wind_speed_kt": (0, 120),
              "wind_gust_kt": (0, 160), "precip_mm": (0, 120),
              "snow_mm": (0, 400), "visibility_km": (0, 20),
              "cloud_ceiling_ft": (0, 45000)}
    n_oob = 0
    for c, (lo, hi) in bounds.items():
        mask = (df[c] < lo) | (df[c] > hi)
        n_oob += int(mask.sum())
        df.loc[mask, c] = np.nan
    log.add("weather: out-of-range values -> NaN", n_oob)

    # rebuild a complete hourly grid per airport, then interpolate short gaps
    full_idx = pd.date_range(df["obs_time_utc"].min(), df["obs_time_utc"].max(),
                             freq="h", tz="UTC")
    out = []
    for code, g in df.groupby("airport_code", sort=False):
        g = g.set_index("obs_time_utc").sort_index().reindex(full_idx)
        g["airport_code"] = code
        g["weather_imputed"] = g[num_cols].isna().all(axis=1).astype("int8")
        g[num_cols] = g[num_cols].interpolate(limit=3, limit_direction="both")
        g["condition"] = g["condition"].ffill(limit=3).fillna("Unknown")
        out.append(g.rename_axis("obs_time_utc").reset_index())
    df = pd.concat(out, ignore_index=True)
    log.add("weather: hours reconstructed by interpolation",
            int(df["weather_imputed"].sum()), "(missing observations)")

    remaining = int(df[num_cols].isna().sum().sum())
    df[num_cols] = df.groupby("airport_code")[num_cols].transform(
        lambda s: s.fillna(s.median()))
    log.add("weather: residual NaN filled with airport median", remaining)
    return df


def clean_flights(df: pd.DataFrame, airports: pd.DataFrame,
                  aircraft: pd.DataFrame, log: QualityLog) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)

    before = len(df)
    df = df.drop_duplicates(subset="flight_id", keep="first")
    log.add("flights: duplicate flight_id removed", before - len(df))

    for c in ("origin", "destination", "carrier_code", "tail_number"):
        df[c] = df[c].astype("string").str.strip().str.upper()

    valid = set(airports["airport_code"])
    bad = ~(df["origin"].isin(valid) & df["destination"].isin(valid))
    log.add("flights: invalid airport codes dropped", int(bad.sum()),
            f"e.g. {sorted(set(df.loc[bad, 'destination'].dropna()))[:4]}")
    df = df[~bad]

    same = df["origin"] == df["destination"]
    log.add("flights: origin == destination dropped", int(same.sum()))
    df = df[~same]

    # ---- timestamps: parse then align local -> UTC using airport tz -------
    tz_map = airports.set_index("airport_code")["timezone"].to_dict()
    dep_local = pd.to_datetime(df["scheduled_departure_local"], errors="coerce")
    arr_local = pd.to_datetime(df["scheduled_arrival_local"], errors="coerce")
    unparsed = int(dep_local.isna().sum() + arr_local.isna().sum())
    log.add("flights: unparseable timestamps", unparsed)

    dep_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    arr_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for code, tz in tz_map.items():
        m = df["origin"] == code
        if m.any():
            dep_utc[m] = (dep_local[m].dt.tz_localize(
                tz, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC"))
        m = df["destination"] == code
        if m.any():
            arr_utc[m] = (arr_local[m].dt.tz_localize(
                tz, ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC"))
    df["scheduled_departure_utc"] = dep_utc
    df["scheduled_arrival_utc"] = arr_utc
    log.add("flights: local timestamps aligned to UTC", len(df),
            f"{len(set(tz_map.values()))} distinct time zones")

    bad_seq = ~(df["scheduled_arrival_utc"] > df["scheduled_departure_utc"])
    log.add("flights: arrival <= departure dropped", int(bad_seq.sum()))
    df = df[~bad_seq]

    # block time sanity (a domestic leg is between 20 min and 9 h)
    block = (df["scheduled_arrival_utc"] - df["scheduled_departure_utc"]).dt.total_seconds() / 60
    bad_block = (block < 20) | (block > 540)
    log.add("flights: implausible block time dropped", int(bad_block.sum()))
    df = df[~bad_block]
    df["scheduled_block_min"] = block[~bad_block]

    # ---- numeric sanity ---------------------------------------------------
    bad_dist = (df["distance_km"] <= 0) | (df["distance_km"] > 8000)
    log.add("flights: invalid distance -> NaN", int(bad_dist.sum()))
    df.loc[bad_dist, "distance_km"] = np.nan
    df["distance_km"] = df["distance_km"].fillna(
        df.groupby(["origin", "destination"])["distance_km"].transform("median"))

    for c in ("departure_delay_min", "arrival_delay_min"):
        bad = (df[c] > MAX_PLAUSIBLE_DELAY_MIN) | (df[c] < MIN_PLAUSIBLE_DELAY_MIN)
        log.add(f"flights: absurd {c} -> NaN", int(bad.sum()))
        df.loc[bad, c] = np.nan

    known_tails = set(aircraft["tail_number"])
    miss_tail = ~df["tail_number"].isin(known_tails)
    log.add("flights: unknown/missing tail number", int(miss_tail.sum()),
            "kept, propagation features flagged")
    df["tail_known"] = (~miss_tail).astype("int8")

    # cancelled flights cannot have a departure delay -> excluded from
    # modelling but preserved for the cancellation analysis
    df["cancelled"] = df["cancelled"].astype("int8")
    df["diverted"] = df["diverted"].astype("int8")
    n_cancel = int(df["cancelled"].sum())
    log.add("flights: cancelled (flagged, not modelled)", n_cancel)

    no_target = df["departure_delay_min"].isna() & (df["cancelled"] == 0)
    log.add("flights: missing target on operated flight dropped", int(no_target.sum()))
    df = df[~no_target]

    log.add("flights: TOTAL rows in -> out", n0, f"-> {len(df):,}")
    return df.sort_values("scheduled_departure_utc").reset_index(drop=True)


# --------------------------------------------------------------------------
def write_report(log: QualityLog) -> None:
    lines = ["# Data Quality Report", "",
             "Automatically generated by `src/data/clean.py`.", "",
             "| Step | Rows affected | Detail |", "|---|---:|---|"]
    for k, v in log.items():
        lines.append(f"| {k} | {v['rows_affected']:,} | {v['detail']} |")
    (REPORTS / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORTS / "data_quality_report.json").write_text(json.dumps(log, indent=2),
                                                      encoding="utf-8")


def main() -> None:
    print("[clean] loading raw sources")
    airports = pd.read_csv(RAW / "airports.csv")
    airlines = pd.read_csv(RAW / "airlines.csv")
    aircraft = pd.read_csv(RAW / "aircraft.csv")
    flights = pd.read_csv(RAW / "flights.csv")
    weather = pd.read_csv(RAW / "weather_hourly.csv")
    holidays = pd.read_csv(RAW / "holiday_calendar.csv")

    log = QualityLog()
    airports = clean_airports(airports, log)
    aircraft = clean_aircraft(aircraft, log)
    weather = clean_weather(weather, airports, log)
    flights = clean_flights(flights, airports, aircraft, log)

    airports.to_parquet(INTERIM / "airports.parquet", index=False)
    airlines.to_parquet(INTERIM / "airlines.parquet", index=False)
    aircraft.to_parquet(INTERIM / "aircraft.parquet", index=False)
    weather.to_parquet(INTERIM / "weather.parquet", index=False)
    flights.to_parquet(INTERIM / "flights.parquet", index=False)
    holidays.to_parquet(INTERIM / "holidays.parquet", index=False)
    write_report(log)
    print(f"[clean] cleaned tables written to {INTERIM}")


if __name__ == "__main__":
    main()
