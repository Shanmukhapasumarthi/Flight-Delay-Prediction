"""Stage 1 - Data Collection.

Produces five INDEPENDENT raw sources that must later be integrated:

    data/raw/flights.csv          operational flight records (local naive times)
    data/raw/airports.csv         geography, runways, capacity, time zone
    data/raw/airlines.csv         carrier reference
    data/raw/aircraft.csv         tail-number level fleet registry
    data/raw/weather_hourly.csv   hourly METAR-style observations in UTC
    data/raw/holiday_calendar.csv US holiday + travel-period calendar

The flight records are simulated, but the simulation encodes real causal
structure so that the downstream modelling is a genuine discovery problem:

  * aircraft fly multi-leg rotations, so delay PROPAGATES down the day
  * airport congestion is endogenous (demand vs. declared hourly capacity)
  * weather severity is derived from raw observations the model never sees
  * carriers differ in operational quality; holidays and peak banks matter

Realistic data-quality defects are injected on purpose (duplicates, missing
weather hours, sentinel values, dirty airport codes, impossible timestamps)
so the cleaning stage has real work to do.
"""
from __future__ import annotations

import argparse
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from src.config import CFG, RAW, SEED
from src.data.reference import (
    AIRCRAFT_TYPE_COLUMNS,
    AIRCRAFT_TYPES,
    AIRLINE_COLUMNS,
    AIRLINES,
    AIRPORT_COLUMNS,
    AIRPORTS,
    CLIMATE_PARAMS,
    FLEET_BY_CARRIER_TYPE,
)

UTC = dt.timezone.utc
EARTH_R_KM = 6371.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la = np.radians(lat)[:, None]
    lo = np.radians(lon)[:, None]
    dla = la - la.T
    dlo = lo - lo.T
    a = np.sin(dla / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlo / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Fast AR(1) noise via IIR filtering - gives weather realistic persistence."""
    eps = rng.normal(0, sigma, n)
    return lfilter([1.0], [1.0, -phi], eps)


# --------------------------------------------------------------------------
# reference tables
# --------------------------------------------------------------------------
def build_airports() -> pd.DataFrame:
    return pd.DataFrame(AIRPORTS, columns=AIRPORT_COLUMNS)


def build_airlines() -> pd.DataFrame:
    df = pd.DataFrame(AIRLINES, columns=AIRLINE_COLUMNS)
    df["hubs"] = df["hubs"].apply(lambda h: "|".join(h))
    return df


def build_aircraft(rng: np.random.Generator, airlines: pd.DataFrame,
                   n_aircraft: int) -> pd.DataFrame:
    types = pd.DataFrame(AIRCRAFT_TYPES, columns=AIRCRAFT_TYPE_COLUMNS)
    rows, used = [], set()
    for _, al in airlines.iterrows():
        n_c = max(2, int(round(al["fleet_share"] * n_aircraft)))
        pool = FLEET_BY_CARRIER_TYPE[al["carrier_type"]]
        for _ in range(n_c):
            while True:
                tail = f"N{rng.integers(100, 999)}{chr(rng.integers(65, 91))}{chr(rng.integers(65, 91))}"
                if tail not in used:
                    used.add(tail)
                    break
            model = pool[int(rng.integers(0, len(pool)))]
            # older airframes at ULCC / regional carriers
            age_shift = {"legacy": 0, "low_cost": -2, "ultra_low_cost": 3,
                         "regional": 5}[al["carrier_type"]]
            age = float(np.clip(rng.gamma(2.6, 3.2) + age_shift, 0.4, 31))
            rows.append({
                "tail_number": tail,
                "carrier_code": al["carrier_code"],
                "aircraft_model": model,
                "manufacture_year": 2024 - int(round(age)),
                "age_years_2024": round(age, 1),
            })
    ac = pd.DataFrame(rows).merge(types, on="aircraft_model", how="left")
    return ac


def build_holiday_calendar(start: dt.date, end: dt.date) -> pd.DataFrame:
    import holidays as hol

    years = range(start.year, end.year + 1)
    us = hol.UnitedStates(years=list(years), observed=True)
    heavy = {"Thanksgiving", "Christmas Day", "New Year's Day",
             "Independence Day", "Memorial Day", "Labor Day"}
    rows = []
    for d, name in sorted(us.items()):
        if not (start <= d <= end + dt.timedelta(days=7)):
            continue
        rows.append({
            "holiday_date": d.isoformat(),
            "holiday_name": name,
            "is_major_travel": int(any(h in name for h in heavy)),
        })
    # extra travel-heavy non-federal days
    for y in years:
        for d, name in [(dt.date(y, 12, 24), "Christmas Eve"),
                        (dt.date(y, 12, 26), "Day after Christmas"),
                        (dt.date(y, 12, 31), "New Year's Eve")]:
            if start <= d <= end:
                rows.append({"holiday_date": d.isoformat(),
                             "holiday_name": name, "is_major_travel": 1})
    df = pd.DataFrame(rows).drop_duplicates("holiday_date").sort_values("holiday_date")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# weather
# --------------------------------------------------------------------------
def build_weather(rng: np.random.Generator, airports: pd.DataFrame,
                  start: dt.datetime, n_hours: int):
    """Hourly synthetic METAR observations + a hidden severity index.

    Returns (dataframe, severity_matrix[n_airports, n_hours]).
    """
    hours = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    doy = hours.dayofyear.to_numpy()
    utc_hour = hours.hour.to_numpy()
    n_days = n_hours // 24

    frames, sev_rows = [], []
    for i, ap in airports.iterrows():
        zone = ap["climate_zone"]
        (mean_t, amp, wet_p, mean_precip, snow_ok, fog_p, storm_peak) = CLIMATE_PARAMS[zone]
        std_off = ZoneInfo(ap["timezone"]).utcoffset(dt.datetime(2024, 1, 15)).total_seconds() / 3600
        local_hour = (utc_hour + std_off) % 24

        # --- temperature: seasonal + diurnal + persistent noise -----------
        seasonal = -amp * np.cos(2 * np.pi * (doy - 15) / 365.25)
        diurnal = 5.0 * np.sin(2 * np.pi * (local_hour - 9.5) / 24)
        temp = (mean_t + seasonal + diurnal
                - 0.0018 * ap["elevation_ft"]
                + ar1(rng, n_hours, 0.93, 1.15))

        # --- precipitation: daily wet/dry then intra-day event blocks -----
        season_boost = 1 + 0.45 * np.cos(2 * np.pi * (np.arange(n_days) + start.timetuple().tm_yday - storm_peak * 30.4) / 365.25)
        wet_day = rng.random(n_days) < np.clip(wet_p * season_boost, 0.01, 0.85)
        precip = np.zeros(n_hours)
        storm = np.zeros(n_hours)
        for d in np.flatnonzero(wet_day):
            dur = int(rng.integers(2, 9))
            h0 = int(rng.integers(0, 24 - min(dur, 23)))
            idx = d * 24 + h0 + np.arange(dur)
            idx = idx[idx < n_hours]
            inten = rng.gamma(1.5, mean_precip / 1.5, len(idx))
            precip[idx] += inten
            # convective storms: warm season, afternoon, continental/subtropical
            month = ((start + dt.timedelta(days=int(d))).month)
            conv_p = 0.30 if zone in ("humid_subtropical", "continental",
                                      "continental_cold", "tropical", "south") else 0.08
            conv_p *= 1.0 if 4 <= month <= 9 else 0.25
            if rng.random() < conv_p and inten.mean() > mean_precip * 0.8:
                storm[idx] = 1.0
                precip[idx] *= 1.8

        # --- snow --------------------------------------------------------
        snow = np.zeros(n_hours)
        if snow_ok:
            cold = temp < 1.0
            snow = np.where(cold, precip * rng.uniform(6, 10), 0.0)
            precip = np.where(cold, precip * 0.65, precip)
            storm = np.where(cold & (snow > 12), 1.0, storm)

        # --- wind --------------------------------------------------------
        wind = np.clip(rng.gamma(2.2, 3.4, n_hours) + 0.9 * ar1(rng, n_hours, 0.88, 1.4), 0, None)
        wind += 6.5 * storm + 1.6 * np.clip(precip, 0, 6)
        wind += np.where(np.isin(zone, ["continental_cold", "continental"]), 2.0, 0.0)
        gust = wind * np.where(rng.random(n_hours) < 0.32, rng.uniform(1.25, 1.85, n_hours), 1.0)

        # --- visibility & ceiling ---------------------------------------
        dewpoint = temp - np.clip(rng.gamma(2.0, 2.2, n_hours) - 2.2 * (precip > 0), 0.2, None)
        fog = (rng.random(n_hours) < fog_p * np.where((local_hour >= 3) & (local_hour <= 9), 3.0, 0.25)) \
              & ((temp - dewpoint) < 2.0)
        vis = 16.0 * np.exp(-0.30 * precip - 0.07 * snow)
        vis = np.where(fog, rng.uniform(0.3, 3.0, n_hours), vis)
        vis = np.clip(vis + rng.normal(0, 0.6, n_hours), 0.1, 16.0)
        ceiling = np.where(precip > 0.5, rng.uniform(400, 3500, n_hours),
                           rng.uniform(3000, 25000, n_hours))
        ceiling = np.where(fog, rng.uniform(100, 700, n_hours), ceiling)

        # --- hidden operational severity (NOT written to disk) -----------
        sev = (0.85 * np.clip(precip / 6.0, 0, 1.7)
               + 1.25 * np.clip((gust - 24) / 26.0, 0, 1.6)
               + 1.05 * np.clip((5.0 - vis) / 5.0, 0, 1.0)
               + 1.55 * np.clip(snow / 18.0, 0, 1.7)
               + 1.15 * storm
               + 0.35 * np.clip((ceiling < 800).astype(float), 0, 1)
               + 0.30 * np.clip((temp - 36) / 8.0, 0, 1)
               + 0.25 * np.clip((-2 - temp) / 12.0, 0, 1))
        sev_rows.append(sev)

        cond = np.where(storm > 0, "Thunderstorm",
               np.where(snow > 0.5, "Snow",
               np.where(precip > 2.5, "Heavy Rain",
               np.where(precip > 0.2, "Rain",
               np.where(fog, "Fog", "Clear")))))

        frames.append(pd.DataFrame({
            "airport_code": ap["airport_code"],
            "obs_time_utc": hours,
            "temperature_c": np.round(temp, 1),
            "dewpoint_c": np.round(dewpoint, 1),
            "wind_speed_kt": np.round(wind, 1),
            "wind_gust_kt": np.round(gust, 1),
            "precip_mm": np.round(precip, 2),
            "snow_mm": np.round(snow, 2),
            "visibility_km": np.round(vis, 2),
            "cloud_ceiling_ft": np.round(ceiling, 0),
            "condition": cond,
        }))

    return pd.concat(frames, ignore_index=True), np.vstack(sev_rows)


# --------------------------------------------------------------------------
# flights
# --------------------------------------------------------------------------
HOUR_PRESSURE = np.array([  # local departure hour -> systemic pressure
    0.10, 0.05, 0.03, 0.03, 0.10, 0.22, 0.30, 0.42, 0.55, 0.62, 0.66, 0.72,
    0.78, 0.84, 0.90, 0.98, 1.06, 1.12, 1.10, 1.00, 0.86, 0.68, 0.45, 0.24])
DOW_EFFECT = np.array([0.11, 0.02, 0.00, 0.07, 0.16, -0.19, -0.05])  # Mon..Sun


def _rotation_schedule(rng, carrier_type):
    """Number of legs an aircraft flies in a day, by carrier type."""
    table = {"legacy": ([2, 3, 4, 5], [0.18, 0.36, 0.32, 0.14]),
             "low_cost": ([3, 4, 5, 6], [0.16, 0.32, 0.32, 0.20]),
             "ultra_low_cost": ([2, 3, 4, 5], [0.22, 0.36, 0.30, 0.12]),
             "regional": ([3, 4, 5, 6], [0.14, 0.30, 0.34, 0.22])}
    legs, p = table[carrier_type]
    return int(rng.choice(legs, p=p))


def build_flights(rng, airports, airlines, aircraft, holiday_df,
                  sev, start_dt, n_days):
    codes = airports["airport_code"].to_list()
    idx_of = {c: i for i, c in enumerate(codes)}
    lat = airports["latitude"].to_numpy()
    lon = airports["longitude"].to_numpy()
    dist = haversine_matrix(lat, lon)
    cap = airports["hourly_capacity"].to_numpy().astype(float)
    tzs = airports["timezone"].to_list()
    n_ap, n_hours = sev.shape

    # local-offset cache: (tz, day) -> utc offset hours
    off_cache: dict[tuple[str, int], float] = {}

    def offset(tz_i: int, day: int) -> float:
        key = (tzs[tz_i], day)
        v = off_cache.get(key)
        if v is None:
            when = start_dt + dt.timedelta(days=day)
            v = ZoneInfo(tzs[tz_i]).utcoffset(when.replace(tzinfo=None)).total_seconds() / 3600
            off_cache[key] = v
        return v

    # carrier-specific destination probability matrices
    dest_p = {}
    for _, al in airlines.iterrows():
        hubs = al["hubs"].split("|")
        pull = cap.copy()
        for h in hubs:
            pull[idx_of[h]] *= 4.5
        w = pull[None, :] * np.exp(-dist / 2400.0)
        np.fill_diagonal(w, 0.0)
        dest_p[al["carrier_code"]] = w / w.sum(axis=1, keepdims=True)

    al_by_code = airlines.set_index("carrier_code")
    ac = aircraft.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    # assign each airframe a home base drawn from its carrier's hubs
    bases = []
    for _, a in ac.iterrows():
        hubs = al_by_code.loc[a["carrier_code"], "hubs"].split("|")
        bases.append(hubs[int(rng.integers(0, len(hubs)))])
    ac["base"] = bases

    holi = pd.to_datetime(holiday_df.loc[holiday_df.is_major_travel == 1, "holiday_date"])
    holi_days = {int((d.date() - start_dt.date()).days) for d in holi}
    holiday_prox = np.zeros(n_days + 5)
    for hd in holi_days:
        for k in range(-3, 4):
            if 0 <= hd + k < len(holiday_prox):
                holiday_prox[hd + k] = max(holiday_prox[hd + k], 1.0 - abs(k) * 0.22)

    # ---------------- pass 1: schedules (no delays yet) -------------------
    rec_tail, rec_car, rec_o, rec_d = [], [], [], []
    rec_dep, rec_arr, rec_dist, rec_leg = [], [], [], []
    rec_turn, rec_fnum = [], []

    for ai, a in ac.iterrows():
        carrier = a["carrier_code"]
        ctype = al_by_code.loc[carrier, "carrier_type"]
        P = dest_p[carrier]
        cur = idx_of[a["base"]]
        seats = a["seats"]
        fnum_base = int(rng.integers(100, 6000))
        for day in range(n_days):
            if rng.random() < 0.045:      # maintenance / spare day
                cur = idx_of[a["base"]]
                continue
            n_legs = _rotation_schedule(rng, ctype)
            start_local = float(np.clip(rng.normal(6.7, 1.15), 4.6, 10.0))
            t_utc = (start_dt + dt.timedelta(days=day)
                     + dt.timedelta(hours=start_local - offset(cur, day)))
            for leg in range(n_legs):
                nxt = int(rng.choice(n_ap, p=P[cur]))
                d_km = float(dist[cur, nxt])
                block = 28.0 + d_km / 11.6 + rng.normal(0, 4)
                block = max(block, 35.0)
                dep_utc = t_utc
                arr_utc = dep_utc + dt.timedelta(minutes=block)
                turn = float(np.clip(32 + seats * 0.11 + rng.gamma(2.0, 7.0), 30, 130))
                rec_tail.append(a["tail_number"]); rec_car.append(carrier)
                rec_o.append(cur); rec_d.append(nxt)
                rec_dep.append(dep_utc); rec_arr.append(arr_utc)
                rec_dist.append(d_km); rec_leg.append(leg)
                rec_turn.append(turn)
                rec_fnum.append(fnum_base + leg + day % 7)
                t_utc = arr_utc + dt.timedelta(minutes=turn)
                cur = nxt
                if t_utc.hour > 22 or leg == n_legs - 1:
                    break
            if rng.random() < 0.55:       # most airframes overnight at base
                cur = idx_of[a["base"]]

    n = len(rec_tail)
    dep_utc = pd.DatetimeIndex(rec_dep)
    arr_utc = pd.DatetimeIndex(rec_arr)
    origin = np.array(rec_o)
    dest = np.array(rec_d)
    distance = np.array(rec_dist)
    turn_min = np.array(rec_turn)
    carrier_arr = np.array(rec_car)
    tail_arr = np.array(rec_tail)

    hour_idx_dep = ((dep_utc - pd.Timestamp(start_dt)) //
                    pd.Timedelta("1h")).to_numpy().astype(int)
    hour_idx_arr = ((arr_utc - pd.Timestamp(start_dt)) //
                    pd.Timedelta("1h")).to_numpy().astype(int)
    hour_idx_dep = np.clip(hour_idx_dep, 0, n_hours - 1)
    hour_idx_arr = np.clip(hour_idx_arr, 0, n_hours - 1)

    # ---------------- endogenous congestion ------------------------------
    demand = np.zeros((n_ap, n_hours))
    np.add.at(demand, (origin, hour_idx_dep), 1.0)
    np.add.at(demand, (dest, hour_idx_arr), 0.85)
    # simulated schedule is a sample of the real network -> scale to realism
    scale = 6.0
    congestion = demand * scale / cap[:, None]
    cong_dep = congestion[origin, hour_idx_dep]
    cong_arr = congestion[dest, hour_idx_arr]

    # ---------------- delay drivers --------------------------------------
    day_idx = np.clip(hour_idx_dep // 24, 0, n_days - 1)
    off_dep = np.array([offset(o, d) for o, d in zip(origin, day_idx)])
    local_dep_hour = ((dep_utc.hour.to_numpy() + off_dep) % 24).astype(int)
    dow = ((dep_utc + pd.to_timedelta(off_dep, unit="h")).dayofweek).to_numpy()

    ops_q = al_by_code["ops_quality_index"].to_dict()
    ops = np.array([ops_q[c] for c in carrier_arr])
    rel_map = aircraft.set_index("tail_number")["reliability_index"].to_dict()
    age_map = aircraft.set_index("tail_number")["age_years_2024"].to_dict()
    rel = np.array([rel_map[t] for t in tail_arr])
    age = np.array([age_map[t] for t in tail_arr])

    wx_o = sev[origin, hour_idx_dep]
    wx_d = sev[dest, hour_idx_arr]

    score = (-2.12
             + ops
             + 1.05 * HOUR_PRESSURE[local_dep_hour]
             + DOW_EFFECT[dow]
             + 1.35 * np.clip(cong_dep - 0.80, 0, 1.1)
             + 0.60 * wx_o
             + 0.20 * wx_d
             + 0.55 * holiday_prox[day_idx]
             + 0.11 * (distance - distance.mean()) / distance.std()
             + 0.013 * age + rel
             + rng.normal(0, 0.35, n))

    p_event = 1.0 / (1.0 + np.exp(-score))
    fires = rng.random(n) < p_event
    magnitude = rng.gamma(1.30, 32.0 * np.exp(0.42 * np.clip(score, -2, 2)), n)
    independent = np.where(fires, magnitude, rng.normal(-2.6, 3.4, n))

    # ---------------- sequential delay propagation ------------------------
    order = np.lexsort((hour_idx_dep, tail_arr))
    dep_delay = np.zeros(n)
    arr_delay = np.zeros(n)
    prev_tail, prev_arr_delay, prev_arr_hr = None, 0.0, -10 ** 9
    min_turn = 30.0
    for i in order:
        t = tail_arr[i]
        if t != prev_tail:
            prev_tail, prev_arr_delay, prev_arr_hr = t, 0.0, -10 ** 9
        gap = hour_idx_dep[i] - prev_arr_hr
        if gap > 10:                       # overnight reset, mostly recovered
            carry = 0.18 * max(prev_arr_delay - 45.0, 0.0)
        else:
            slack = max(turn_min[i] - min_turn, 0.0)
            carry = max(prev_arr_delay - slack, 0.0) * 0.94
        d = max(carry + independent[i], -22.0)
        dep_delay[i] = d
        air = (-0.11 * d + 0.85 * wx_d[i] * 6.0
               + 0.9 * np.clip(cong_arr[i] - 0.9, 0, 1.2) * 8.0
               + rng.normal(0, 7.0))
        a_del = d + air
        arr_delay[i] = a_del
        prev_arr_delay, prev_arr_hr = a_del, hour_idx_arr[i]

    # ---------------- cancellations & diversions --------------------------
    p_cancel = 1 / (1 + np.exp(-(-5.05 + 1.45 * wx_o + 0.55 * wx_d + 0.35 * (score > 1.0))))
    cancelled = rng.random(n) < p_cancel
    diverted = (~cancelled) & (rng.random(n) < 1 / (1 + np.exp(-(-7.4 + 1.5 * wx_d))))

    dep_local = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    arr_local = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    for i_ap in range(n_ap):
        tz = tzs[i_ap]
        m_o = origin == i_ap
        if m_o.any():
            dep_local[m_o] = dep_utc[m_o].tz_convert(tz).tz_localize(None)
        m_d = dest == i_ap
        if m_d.any():
            arr_local[m_d] = arr_utc[m_d].tz_convert(tz).tz_localize(None)

    flights = pd.DataFrame({
        "flight_id": [f"FL{i:08d}" for i in range(n)],
        "carrier_code": carrier_arr,
        "flight_number": rec_fnum,
        "tail_number": tail_arr,
        "origin": [codes[i] for i in origin],
        "destination": [codes[i] for i in dest],
        "scheduled_departure_local": dep_local.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "scheduled_arrival_local": arr_local.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "scheduled_elapsed_min": np.round((arr_utc - dep_utc) / pd.Timedelta("1min"), 0),
        "distance_km": np.round(distance, 1),
        "scheduled_turnaround_min": np.round(turn_min, 0),
        "leg_number": rec_leg,
        "departure_delay_min": np.round(dep_delay, 0),
        "arrival_delay_min": np.round(arr_delay, 0),
        "cancelled": cancelled.astype(int),
        "diverted": diverted.astype(int),
    })
    flights.loc[flights.cancelled == 1, ["departure_delay_min", "arrival_delay_min"]] = np.nan
    return flights.sort_values("scheduled_departure_local").reset_index(drop=True)


# --------------------------------------------------------------------------
# data-quality defects (deliberate)
# --------------------------------------------------------------------------
def inject_defects(rng, flights: pd.DataFrame, weather: pd.DataFrame):
    f = flights.copy()

    # 1. exact duplicate flight rows (double ingestion)
    dupes = f.sample(frac=0.004, random_state=SEED)
    f = pd.concat([f, dupes], ignore_index=True)

    # 2. dirty airport codes: whitespace / lower case
    m = f.sample(frac=0.012, random_state=SEED + 1).index
    f.loc[m, "origin"] = " " + f.loc[m, "origin"].str.lower() + " "

    # 3. invalid airport codes
    m = f.sample(frac=0.0015, random_state=SEED + 2).index
    f.loc[m, "destination"] = rng.choice(["ZZZ", "N/A", "---", ""], size=len(m))

    # 4. impossible timestamps (arrival before departure)
    m = f.sample(frac=0.001, random_state=SEED + 3).index
    f.loc[m, "scheduled_arrival_local"] = f.loc[m, "scheduled_departure_local"]

    # 5. sentinel / impossible values
    m = f.sample(frac=0.002, random_state=SEED + 4).index
    f.loc[m, "distance_km"] = -999.0
    m = f.sample(frac=0.0008, random_state=SEED + 5).index
    f.loc[m, "departure_delay_min"] = 99999

    # 6. missing tail numbers
    m = f.sample(frac=0.006, random_state=SEED + 6).index
    f.loc[m, "tail_number"] = None

    w = weather.copy()
    # 7. 2.2% of hourly observations simply never arrived
    keep = rng.random(len(w)) > 0.022
    w = w.loc[keep].reset_index(drop=True)
    # 8. sentinel -999 for unreported elements
    for col, frac in [("visibility_km", 0.006), ("wind_gust_kt", 0.010),
                      ("cloud_ceiling_ft", 0.008)]:
        m = w.sample(frac=frac, random_state=SEED + 7).index
        w.loc[m, col] = -999.0
    # 9. duplicated observations
    w = pd.concat([w, w.sample(frac=0.002, random_state=SEED + 8)], ignore_index=True)
    return f.sample(frac=1.0, random_state=SEED).reset_index(drop=True), w


# --------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate raw multi-source data")
    ap.add_argument("--n-aircraft", type=int, default=CFG["data"]["n_aircraft"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    start = dt.datetime.fromisoformat(CFG["data"]["start_date"])
    end = dt.datetime.fromisoformat(CFG["data"]["end_date"])
    n_days = (end - start).days + 1
    n_hours = n_days * 24

    print(f"[collect] window {start.date()} .. {end.date()}  ({n_days} days)")
    airports = build_airports()
    airlines = build_airlines()
    aircraft = build_aircraft(rng, airlines, args.n_aircraft)
    holiday_df = build_holiday_calendar(start.date(), end.date())
    print(f"[collect] {len(airports)} airports, {len(airlines)} carriers, "
          f"{len(aircraft)} airframes, {len(holiday_df)} holiday dates")

    weather, sev = build_weather(rng, airports, start.replace(tzinfo=UTC), n_hours)
    print(f"[collect] weather observations: {len(weather):,}")

    flights = build_flights(rng, airports, airlines, aircraft, holiday_df,
                            sev, start.replace(tzinfo=UTC), n_days)
    rate = (flights.departure_delay_min >= 15).mean()
    print(f"[collect] flights: {len(flights):,}   delay>=15min rate: {rate:.1%}   "
          f"cancelled: {flights.cancelled.mean():.2%}")

    flights, weather = inject_defects(rng, flights, weather)

    airports.to_csv(RAW / "airports.csv", index=False)
    airlines.to_csv(RAW / "airlines.csv", index=False)
    aircraft.to_csv(RAW / "aircraft.csv", index=False)
    holiday_df.to_csv(RAW / "holiday_calendar.csv", index=False)
    flights.to_csv(RAW / "flights.csv", index=False)
    weather.to_csv(RAW / "weather_hourly.csv", index=False)
    print(f"[collect] wrote 6 raw files to {RAW}")


if __name__ == "__main__":
    main()
