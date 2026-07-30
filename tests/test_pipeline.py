"""Test suite.

The tests that matter most here are the LEAKAGE tests: a flight-delay model
is trivially easy to make look excellent by accident, and the usual culprits
are (a) rolling features that include the current day and (b) a random split
that lets the future inform the past.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_flights, clean_weather, QualityLog
from src.data.generate_synthetic import haversine_matrix
from src.features.build_features import (add_geo_features, add_propagation_features,
                                         add_time_features)
from src.models.featurize_online import haversine, wx_severity


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def test_haversine_known_distance():
    # JFK -> LAX is ~3,970 km
    d = haversine(40.6413, -73.7781, 33.9416, -118.4085)
    assert 3900 < d < 4050


def test_haversine_matrix_symmetry():
    lat = np.array([33.6, 41.9, 25.8])
    lon = np.array([-84.4, -87.9, -80.3])
    m = haversine_matrix(lat, lon)
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0)


# --------------------------------------------------------------------------
# weather severity
# --------------------------------------------------------------------------
def test_severity_zero_in_clear_conditions():
    assert wx_severity(precip=0, gust=5, vis=16, snow=0, storm=0, ceiling=20000) == 0


def test_severity_monotonic_in_precipitation():
    a = wx_severity(1, 5, 16, 0, 0, 20000)
    b = wx_severity(8, 5, 16, 0, 0, 20000)
    assert b > a


def test_storm_dominates_clear():
    clear = wx_severity(0, 5, 16, 0, 0, 20000)
    storm = wx_severity(12, 45, 1.2, 0, 1, 400)
    assert storm > clear + 3


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------
@pytest.fixture
def airports():
    return pd.DataFrame({
        "airport_code": ["ORD", "ATL"],
        "timezone": ["America/Chicago", "America/New_York"],
    })


@pytest.fixture
def aircraft():
    return pd.DataFrame({"tail_number": ["N123AB"]})


def _flight_rows():
    return pd.DataFrame({
        "flight_id": ["F1", "F1", "F2", "F3", "F4"],           # F1 duplicated
        "carrier_code": ["AA"] * 5,
        "tail_number": ["N123AB"] * 5,
        "origin": [" ord ", " ord ", "ORD", "ZZZ", "ORD"],      # dirty + invalid
        "destination": ["ATL", "ATL", "ATL", "ATL", "ATL"],
        "scheduled_departure_local": ["2024-01-05 08:00:00"] * 5,
        "scheduled_arrival_local": ["2024-01-05 10:45:00", "2024-01-05 10:45:00",
                                    "2024-01-05 07:00:00",   # arrival before departure
                                    "2024-01-05 10:45:00", "2024-01-05 10:45:00"],
        "distance_km": [975, 975, 975, 975, -999],             # sentinel
        "departure_delay_min": [12, 12, 5, 5, 99999],          # absurd value
        "arrival_delay_min": [8, 8, 2, 2, 3],
        "cancelled": [0, 0, 0, 0, 0],
        "diverted": [0] * 5,
    })


def test_cleaning_removes_duplicates_and_bad_codes(airports, aircraft):
    out = clean_flights(_flight_rows(), airports, aircraft, QualityLog())
    assert "F1" in set(out["flight_id"])
    assert (out["flight_id"] == "F1").sum() == 1, "duplicate flight_id survived"
    assert "ZZZ" not in set(out["origin"]), "invalid airport code survived"


def test_cleaning_normalises_dirty_codes(airports, aircraft):
    out = clean_flights(_flight_rows(), airports, aircraft, QualityLog())
    assert set(out["origin"]) <= {"ORD", "ATL"}


def test_cleaning_drops_impossible_timestamps(airports, aircraft):
    out = clean_flights(_flight_rows(), airports, aircraft, QualityLog())
    assert (out["scheduled_arrival_utc"] > out["scheduled_departure_utc"]).all()


def test_cleaning_nullifies_sentinels(airports, aircraft):
    out = clean_flights(_flight_rows(), airports, aircraft, QualityLog())
    assert not (out["distance_km"] == -999).any()
    assert not (out["departure_delay_min"] > 1440).any()


def test_timezone_alignment_is_applied(airports, aircraft):
    """08:00 in Chicago is 14:00 UTC in January -- not 08:00 UTC."""
    out = clean_flights(_flight_rows(), airports, aircraft, QualityLog())
    dep = out.loc[out["origin"] == "ORD", "scheduled_departure_utc"].iloc[0]
    assert dep.hour == 14, f"expected 14:00 UTC, got {dep}"


# --------------------------------------------------------------------------
# feature engineering
# --------------------------------------------------------------------------
def _feature_frame():
    n = 48
    base = pd.Timestamp("2024-01-01 00:00:00")
    return pd.DataFrame({
        "scheduled_departure_local": [base + pd.Timedelta(hours=i) for i in range(n)],
        "days_from_major_holiday": np.linspace(-6, 6, n),
        "origin_latitude": 41.97, "origin_longitude": -87.9,
        "dest_latitude": 33.64, "dest_longitude": -84.43,
        "origin_region": "Midwest", "dest_region": "Southeast",
        "origin_is_hub": 1, "dest_is_hub": 1,
        "origin_elevation_ft": 672, "dest_elevation_ft": 1026,
    })


def test_cyclical_time_encoding_wraps():
    f = add_time_features(_feature_frame())
    h0 = f.loc[f["dep_hour"] == 0, ["hour_sin", "hour_cos"]].iloc[0]
    h23 = f.loc[f["dep_hour"] == 23, ["hour_sin", "hour_cos"]].iloc[0]
    dist = np.hypot(h0["hour_sin"] - h23["hour_sin"], h0["hour_cos"] - h23["hour_cos"])
    assert dist < 0.3, "23:00 and 00:00 should be adjacent in cyclical space"


def test_weekend_flag_matches_calendar():
    f = add_time_features(_feature_frame())
    assert f.loc[f["day_of_week"] >= 5, "is_weekend"].eq(1).all()
    assert f.loc[f["day_of_week"] < 5, "is_weekend"].eq(0).all()


def test_geo_bearing_northbound_is_sane():
    f = _feature_frame()
    f["dest_latitude"] = 47.45      # push destination well north
    f["dest_longitude"] = -87.9     # same meridian
    f = add_geo_features(f)
    assert f["route_bearing_deg"].between(-1, 1).all() or f["route_bearing_deg"].between(359, 360).all()
    assert (f["lat_delta"] > 0).all()


# --------------------------------------------------------------------------
# LEAKAGE -- the tests that actually protect the result
# --------------------------------------------------------------------------
def test_propagation_uses_only_landed_inbound_legs():
    """prev_leg_arr_delay_min must be NaN when the inbound leg had not landed."""
    df = pd.DataFrame({
        "tail_number": ["N1", "N1"],
        "scheduled_departure_utc": pd.to_datetime(
            ["2024-01-01 08:00", "2024-01-01 10:00"], utc=True),
        "scheduled_arrival_utc": pd.to_datetime(
            ["2024-01-01 09:30", "2024-01-01 12:00"], utc=True),
        "origin": ["ORD", "ATL"], "destination": ["ATL", "MIA"],
        # inbound leg lands 60 min late -> 10:30, AFTER the 10:00 departure
        "departure_delay_min": [45.0, 20.0],
        "arrival_delay_min": [60.0, 25.0],
        "cancelled": [0, 0],
        "local_departure_date": [dt.date(2024, 1, 1)] * 2,
    })
    out = add_propagation_features(df).sort_values("scheduled_departure_utc")
    second = out.iloc[1]
    assert np.isnan(second["prev_leg_arr_delay_min"]), \
        "used an arrival delay that was not yet observable at prediction time"
    assert second["prev_leg_known"] == 0
    # falls back to the inbound leg's DEPARTURE delay, which was observable
    assert second["upstream_delay_min"] == pytest.approx(45.0)


def test_propagation_uses_arrival_when_it_did_land_in_time():
    df = pd.DataFrame({
        "tail_number": ["N1", "N1"],
        "scheduled_departure_utc": pd.to_datetime(
            ["2024-01-01 08:00", "2024-01-01 12:00"], utc=True),
        "scheduled_arrival_utc": pd.to_datetime(
            ["2024-01-01 09:30", "2024-01-01 14:00"], utc=True),
        "origin": ["ORD", "ATL"], "destination": ["ATL", "MIA"],
        "departure_delay_min": [45.0, 20.0],
        "arrival_delay_min": [60.0, 25.0],       # lands 10:30, before the 12:00 push
        "cancelled": [0, 0],
        "local_departure_date": [dt.date(2024, 1, 1)] * 2,
    })
    out = add_propagation_features(df).sort_values("scheduled_departure_utc")
    second = out.iloc[1]
    assert second["prev_leg_known"] == 1
    assert second["upstream_delay_min"] == pytest.approx(60.0)


def test_chain_breaks_when_aircraft_did_not_fly_that_route():
    """If the previous leg ended somewhere else, there is no rotation link."""
    df = pd.DataFrame({
        "tail_number": ["N1", "N1"],
        "scheduled_departure_utc": pd.to_datetime(
            ["2024-01-01 08:00", "2024-01-01 12:00"], utc=True),
        "scheduled_arrival_utc": pd.to_datetime(
            ["2024-01-01 09:30", "2024-01-01 14:00"], utc=True),
        "origin": ["ORD", "DEN"],                # previous leg landed at ATL
        "destination": ["ATL", "MIA"],
        "departure_delay_min": [45.0, 20.0],
        "arrival_delay_min": [60.0, 25.0],
        "cancelled": [0, 0],
        "local_departure_date": [dt.date(2024, 1, 1)] * 2,
    })
    out = add_propagation_features(df).sort_values("scheduled_departure_utc")
    assert out.iloc[1]["upstream_delay_min"] == 0.0


def test_split_is_chronological_and_disjoint():
    from src.models.dataset import make_splits
    sp = make_splits()
    tr = sp.meta_train["date"].max()
    va_lo, va_hi = sp.meta_valid["date"].min(), sp.meta_valid["date"].max()
    te = sp.meta_test["date"].min()
    assert tr < va_lo, "train overlaps validation"
    assert va_hi < te, "validation overlaps test"


def test_no_target_derived_column_in_feature_set():
    from src.models.dataset import ALL_FEATURES
    banned = {"departure_delay_min", "arrival_delay_min", "is_delayed", "cancelled"}
    assert not (set(ALL_FEATURES) & banned)


# --------------------------------------------------------------------------
# online / offline parity and the API
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    pytest.importorskip("httpx2")
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["n_features"] > 0


def test_predict_returns_calibrated_probability(client):
    r = client.post("/predict", json={
        "carrier_code": "DL", "origin": "ATL", "destination": "MCO",
        "scheduled_departure_local": "2024-04-16T07:05:00"})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert body["risk_band"] in {"low", "moderate", "elevated", "high"}


def test_bad_weather_scores_higher_than_clear(client):
    base = {"carrier_code": "AA", "origin": "ORD", "destination": "LGA",
            "scheduled_departure_local": "2024-04-12T18:00:00"}
    clear = client.post("/predict", json={**base, "origin_weather": {
        "precip_mm": 0, "wind_gust_kt": 5, "visibility_km": 16}}).json()
    storm = client.post("/predict", json={**base, "origin_weather": {
        "precip_mm": 14, "wind_gust_kt": 45, "visibility_km": 1.0,
        "cloud_ceiling_ft": 400, "condition": "Thunderstorm"}}).json()
    assert storm["delay_probability"] > clear["delay_probability"]


def test_late_inbound_scores_higher(client):
    base = {"carrier_code": "AA", "origin": "ORD", "destination": "LGA",
            "scheduled_departure_local": "2024-04-12T18:00:00",
            "scheduled_ground_time_min": 45}
    on_time = client.post("/predict", json={**base, "prev_leg_arr_delay_min": 0}).json()
    late = client.post("/predict", json={**base, "prev_leg_arr_delay_min": 120}).json()
    assert late["delay_probability"] > on_time["delay_probability"] + 0.1


def test_unknown_airport_is_rejected(client):
    r = client.post("/predict", json={
        "carrier_code": "AA", "origin": "XXX", "destination": "ORD",
        "scheduled_departure_local": "2024-04-12T10:00:00"})
    assert r.status_code == 422


def test_batch_watchlist_is_sorted(client):
    r = client.post("/delay-risk", json={"flights": [
        {"carrier_code": "WN", "origin": "MDW", "destination": "DAL",
         "scheduled_departure_local": "2024-04-12T09:00:00"},
        {"carrier_code": "AA", "origin": "DFW", "destination": "ORD",
         "scheduled_departure_local": "2024-04-12T17:00:00",
         "prev_leg_arr_delay_min": 95, "scheduled_ground_time_min": 40},
    ], "explain": False})
    assert r.status_code == 200
    probs = [p["delay_probability"] for p in r.json()["all"]]
    assert probs == sorted(probs, reverse=True)
