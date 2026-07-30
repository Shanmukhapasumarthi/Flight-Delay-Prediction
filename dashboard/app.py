from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FIGURES, MODELS_DIR, PROCESSED, REPORTS  # noqa: E402

st.set_page_config(page_title="Flight Delay Intelligence", page_icon="✈",
                   layout="wide", initial_sidebar_state="expanded")

ACCENT = "#c8553d"

@st.cache_data(show_spinner=False)
def load_flights(n: int = 120_000) -> pd.DataFrame:
    cols = ["date", "carrier_code", "carrier_name", "origin", "destination", "route",
            "is_delayed", "departure_delay_min", "dep_hour", "day_of_week", "month",
            "dep_wx_condition", "dep_rain_intensity", "dep_wind_category",
            "dep_visibility_level", "dep_wx_severity", "origin_congestion_ratio",
            "upstream_delay_min", "in_holiday_window", "is_peak_hour",
            "origin_latitude", "origin_longitude", "leg_depth"]
    df = pd.read_parquet(PROCESSED / "features.parquet", columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df.sample(min(n, len(df)), random_state=0) if len(df) > n else df


@st.cache_data(show_spinner=False)
def load_report(name: str):
    p = REPORTS / name
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text())
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return p.read_text()


@st.cache_resource(show_spinner=False)
def load_model():
    b = joblib.load(MODELS_DIR / "final_model.joblib")
    from src.models.explain import get_explainer
    return b, get_explainer(b)


@st.cache_resource(show_spinner=False)
def load_store():
    from src.models.featurize_online import load_store as _ls
    return _ls()


def fig(name: str, caption: str = "") -> None:
    p = FIGURES / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not generated yet: {name}")


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    for col, (label, value, delta) in zip(st.columns(len(items)), items):
        col.metric(label, value, delta)


df = load_flights()
econ = load_report("business_economics.json")
eda = load_report("eda_facts.json")
metrics = load_report("model_metrics.json")

st.sidebar.title("✈ Delay Intelligence")
view = st.sidebar.radio("View", ["Overview", "Risk predictor", "Airports", "Airlines",
                                 "Weather", "Model", "Economics"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(
    f"**{len(df):,}** flights sampled  \n"
    f"{df['date'].min():%b %Y} – {df['date'].max():%b %Y}  \n"
    f"{df['origin'].nunique()} airports · {df['carrier_code'].nunique()} carriers")

d_min, d_max = df["date"].min().date(), df["date"].max().date()
rng = st.sidebar.date_input("Date range", (d_min, d_max), min_value=d_min, max_value=d_max)
if isinstance(rng, tuple) and len(rng) == 2:
    df = df[(df["date"].dt.date >= rng[0]) & (df["date"].dt.date <= rng[1])]
carriers = st.sidebar.multiselect("Carriers", sorted(df["carrier_code"].unique()))
if carriers:
    df = df[df["carrier_code"].isin(carriers)]


# --------------------------------------------------------------------------
if view == "Overview":
    st.title("Network overview")
    delayed = df["is_delayed"].mean()
    dm = df.loc[df.is_delayed == 1, "departure_delay_min"]
    kpi_row([
        ("Flights", f"{len(df):,}", None),
        ("Delay rate (≥15 min)", f"{delayed:.1%}", None),
        ("Mean delay when late", f"{dm.mean():.0f} min", None),
        ("Total delay hours", f"{df['departure_delay_min'].clip(lower=0).sum()/60:,.0f}", None),
        ("Worst hour", f"{df.groupby('dep_hour')['is_delayed'].mean().idxmax():02d}:00", None),
    ])
    st.divider()
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Daily delay rate")
        daily = df.groupby(df["date"].dt.date)["is_delayed"].mean()
        st.line_chart(daily, height=280, color=ACCENT)
    with c2:
        st.subheader("By departure hour")
        st.bar_chart(df.groupby("dep_hour")["is_delayed"].mean(), height=280, color=ACCENT)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("By weekday")
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        s = df.groupby("day_of_week")["is_delayed"].mean()
        s.index = [names[i] for i in s.index]
        st.bar_chart(s, height=250, color=ACCENT)
    with c4:
        st.subheader("By rotation depth (propagation)")
        s = df.groupby("leg_depth")["is_delayed"].mean()
        st.bar_chart(s[s.index <= 5], height=250, color=ACCENT)
    if eda:
        st.info(f"Delay risk compounds through the day: **{eda['early_hour_rate']:.1%}** at "
                f"06:00 rises to **{eda['peak_hour_rate']:.1%}** at "
                f"{eda['peak_hour']:02d}:00. Worst slot of the week: **{eda['worst_dow_hour']}**.")

elif view == "Risk predictor":
    st.title("Score a flight")
    st.caption("Live scoring through the same code path the API uses.")
    try:
        bundle, explainer = load_model()
        store = load_store()
    except Exception as e:
        st.error(f"Model not available: {e}")
        st.stop()

    from src.models.explain import explain_rows, narrate
    from src.models.featurize_online import featurize, to_frame

    airports = sorted(store["airports"])
    c1, c2, c3 = st.columns(3)
    with c1:
        carrier = st.selectbox("Carrier", sorted(store["carrier_type"]))
        origin = st.selectbox("Origin", airports, index=airports.index("ORD"))
        dest = st.selectbox("Destination", airports, index=airports.index("LGA"))
    with c2:
        date = st.date_input("Departure date", pd.Timestamp("2024-04-12"))
        time = st.time_input("Departure time (local)", pd.Timestamp("2024-04-12 18:20").time())
        ground = st.slider("Scheduled ground time (min)", 20, 180, 45)
    with c3:
        inbound = st.slider("Inbound aircraft delay (min)", 0, 240, 45)
        precip = st.slider("Precipitation (mm/h)", 0.0, 25.0, 0.0, 0.5)
        gust = st.slider("Wind gust (kt)", 0, 60, 8)
        vis = st.slider("Visibility (km)", 0.2, 16.0, 16.0, 0.2)

    req = {
        "carrier_code": carrier, "origin": origin, "destination": dest,
        "scheduled_departure_local": f"{date}T{time}",
        "scheduled_ground_time_min": ground, "prev_leg_arr_delay_min": inbound,
        "origin_weather": {"precip_mm": precip, "wind_gust_kt": gust,
                           "visibility_km": vis,
                           "condition": "Thunderstorm" if (precip > 5 and gust > 30) else "Rain" if precip > 0.2 else "Clear"},
    }
    try:
        feats = featurize(req, store)
        X = to_frame(feats, bundle["features"], bundle["categories"])
        p = float(bundle["calibrator"].predict_proba(X)[:, 1][0])
        thr = json.loads((MODELS_DIR / "operating_threshold.json").read_text())["threshold"] \
            if (MODELS_DIR / "operating_threshold.json").exists() else 0.5
        contrib = explain_rows(bundle, X, top_n=6, explainer=explainer)[0]

        st.divider()
        a, b = st.columns([1, 2])
        with a:
            band = "HIGH" if p >= 0.5 else "ELEVATED" if p >= 0.3 else "MODERATE" if p >= 0.15 else "LOW"
            st.metric("Delay probability", f"{p:.0%}", band)
            st.metric("Expected delay", f"{p*44:.0f} min")
            st.metric("Action" if p >= thr else "No action",
                      "INTERVENE" if p >= thr else "monitor",
                      f"threshold {thr:.2f}")
        with b:
            st.subheader("Why")
            cd = pd.DataFrame([{"factor": f"{c['label']} ({c['display_value']})",
                                "contribution": c["shap_value"]} for c in contrib])
            st.bar_chart(cd.set_index("factor")["contribution"], horizontal=True,
                         height=260, color=ACCENT)
            st.caption(narrate(contrib, p))
    except Exception as e:
        st.error(f"Could not score this flight: {e}")

elif view == "Airports":
    st.title("Station performance")
    ap = load_report("airport_performance.csv")
    g = (df.groupby("origin").agg(flights=("is_delayed", "size"),
                                  delay_rate=("is_delayed", "mean"),
                                  delay_min=("departure_delay_min", lambda s: s.clip(lower=0).sum()))
           .reset_index())
    g["share_of_delay_minutes"] = g["delay_min"] / g["delay_min"].sum()
    kpi_row([("Stations", f"{len(g)}", None),
             ("Worst by rate", g.sort_values('delay_rate').iloc[-1]['origin'], 
              f"{g['delay_rate'].max():.1%}"),
             ("Largest contributor", g.sort_values('share_of_delay_minutes').iloc[-1]['origin'],
              f"{g['share_of_delay_minutes'].max():.1%} of delay minutes")])
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Delay rate by station")
        st.bar_chart(g.set_index("origin")["delay_rate"].sort_values(ascending=False).head(20),
                     height=380, color=ACCENT)
    with c2:
        st.subheader("Share of network delay minutes")
        st.bar_chart(g.set_index("origin")["share_of_delay_minutes"]
                     .sort_values(ascending=False).head(20), height=380, color="#2f6f8f")
    fig("07_geographic_map.png", "Bubble size = departures, colour = delay rate")
    st.dataframe(g.sort_values("share_of_delay_minutes", ascending=False)
                 .style.format({"delay_rate": "{:.1%}", "share_of_delay_minutes": "{:.2%}",
                                "flights": "{:,.0f}", "delay_min": "{:,.0f}"}),
                 use_container_width=True, height=300)

elif view == "Airlines":
    st.title("Carrier performance")
    g = (df.groupby(["carrier_code", "carrier_name"])
           .agg(flights=("is_delayed", "size"), delay_rate=("is_delayed", "mean"),
                mean_delay=("departure_delay_min", "mean"))
           .reset_index().sort_values("delay_rate"))
    kpi_row([("Best", g.iloc[0]["carrier_code"], f"{g.iloc[0]['delay_rate']:.1%}"),
             ("Worst", g.iloc[-1]["carrier_code"], f"{g.iloc[-1]['delay_rate']:.1%}"),
             ("Spread", f"{(g.iloc[-1]['delay_rate']-g.iloc[0]['delay_rate'])*100:.1f} pp", None)])
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Overall delay rate")
        st.bar_chart(g.set_index("carrier_code")["delay_rate"], height=340, color=ACCENT)
    with c2:
        st.subheader("Peak hours only (07-09, 17-19)")
        pk = df[df["is_peak_hour"] == 1].groupby("carrier_code")["is_delayed"].mean()
        st.bar_chart(pk.sort_values(), height=340, color="#2f6f8f")
    fig("02_delay_by_carrier.png")
    st.dataframe(g.style.format({"delay_rate": "{:.1%}", "mean_delay": "{:.1f}",
                                 "flights": "{:,.0f}"}), use_container_width=True)

elif view == "Weather":
    st.title("Weather impact")
    base = df["is_delayed"].mean()
    storm = df[df["dep_wx_condition"] == "Thunderstorm"]["is_delayed"].mean()
    kpi_row([("Baseline delay rate", f"{base:.1%}", None),
             ("In thunderstorms", f"{storm:.1%}", f"{storm/base:.1f}x"),
             ("Flights in adverse wx", f"{(df['dep_wx_severity']>0.3).mean():.1%}", None)])
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By reported condition")
        st.bar_chart(df.groupby("dep_wx_condition")["is_delayed"].mean().sort_values(),
                     height=300, color=ACCENT)
        st.subheader("By visibility band")
        st.bar_chart(df.groupby("dep_visibility_level", observed=True)["is_delayed"].mean(),
                     height=280, color=ACCENT)
    with c2:
        st.subheader("By precipitation intensity")
        st.bar_chart(df.groupby("dep_rain_intensity", observed=True)["is_delayed"].mean(),
                     height=300, color="#2f6f8f")
        st.subheader("By wind / gust category")
        st.bar_chart(df.groupby("dep_wind_category", observed=True)["is_delayed"].mean(),
                     height=280, color="#2f6f8f")
    fig("06_weather_and_congestion.png")
    st.warning("Weather is a sharp but narrow driver: severe conditions multiply risk, "
               "but they affect a small minority of flights. In the SHAP attribution it "
               "accounts for roughly 7% of total explanatory power — far less than "
               "operational propagation.")

elif view == "Model":
    st.title("Model performance")
    if metrics:
        rows = {k: v for k, v in metrics.items() if not k.startswith("_")}
        t = pd.DataFrame({k: {"valid ROC-AUC": v["valid"]["roc_auc"],
                              "valid PR-AUC": v["valid"]["pr_auc"],
                              "valid Brier": v["valid"]["brier"],
                              "test ROC-AUC": v["test"]["roc_auc"],
                              "test PR-AUC": v["test"]["pr_auc"],
                              "fit (s)": v["fit_seconds"]} for k, v in rows.items()}).T
        t = t.sort_values("valid PR-AUC", ascending=False)
        kpi_row([("Best model", metrics.get("_best_model", "-"), None),
                 ("Test ROC-AUC", f"{t['test ROC-AUC'].max():.3f}", None),
                 ("Candidates compared", f"{len(t)}", None)])
        st.dataframe(t.style.format("{:.4f}", subset=t.columns[:-1])
                     .format("{:.0f}", subset=["fit (s)"]), use_container_width=True)
    tabs = st.tabs(["Comparison", "Calibration", "SHAP global", "SHAP local",
                    "Dependence", "Feature selection", "Tuning"])
    for tab, name, cap in zip(
            tabs,
            ["11_model_comparison.png", "13_calibration.png", "14_shap_global.png",
             "17_shap_local_example.png", "16_shap_dependence.png",
             "10_feature_selection.png", "12_hyperparameter_search.png"],
            ["ROC, PR and calibration across five candidates",
             "Raw vs isotonic — the raw model was already well calibrated",
             "Which factors move predictions network-wide",
             "Why one specific flight was flagged",
             "Model response to its strongest drivers",
             "Consensus of MI, gain and RFE",
             "TPE vs random search"]):
        with tab:
            fig(name, cap)

elif view == "Economics":
    st.title("Alerting economics")
    sweep = load_report("threshold_sweep.csv")
    dec = load_report("risk_deciles.csv")
    if sweep is None or econ is None:
        st.info("Run `python -m src.insights.business` first.")
        st.stop()

    cm = econ["cost_model"]
    st.caption("Every alert costs money and every missed delay costs more. "
               "The threshold is where those two curves cross.")
    c1, c2 = st.columns([1, 3])
    with c1:
        cost = st.number_input("Intervention cost ($)", 100, 5000,
                               int(cm["mitigation_cost_usd"]), 100)
        eff = st.slider("Delay minutes recovered", 0.05, 0.80,
                        float(cm["mitigation_effectiveness"]), 0.05)
        avg = cm["avg_delay_minutes_when_delayed"]
        per_min = cm["cost_per_delay_minute_usd"]
        benefit = avg * eff * per_min
        st.metric("Benefit per correct alert", f"${benefit:,.0f}")
        st.metric("Break-even probability", f"{cost/benefit:.2f}")
    with c2:
        s = sweep.copy()
        s["net"] = s["tp"] * benefit - (s["tp"] + s["fp"]) * cost
        best = s.loc[s["net"].idxmax()]
        st.subheader("Net benefit across the threshold range")
        st.line_chart(s.set_index("threshold")["net"], height=300, color=ACCENT)
        kpi_row([("Optimal threshold", f"{best['threshold']:.2f}", None),
                 ("Alert on", f"{best['alert_rate']:.1%}", "of flights"),
                 ("Precision", f"{best['precision']:.0%}", None),
                 ("Recall", f"{best['recall']:.0%}", None),
                 ("Net benefit", f"${best['net']/1e6:.2f}M", "test window")])
    st.divider()
    if dec is not None:
        c3, c4 = st.columns(2)
        dec = dec.rename(columns={dec.columns[0]: "decile"})
        with c3:
            st.subheader("Delay minutes by risk decile")
            st.bar_chart(dec.set_index("decile")["share_of_delay_minutes"],
                         height=300, color=ACCENT)
        with c4:
            st.subheader("Lift over base rate")
            st.bar_chart(dec.set_index("decile")["lift"], height=300, color="#2f6f8f")
        st.dataframe(dec.style.format({"delay_rate": "{:.1%}", "lift": "{:.2f}",
                                       "share_of_delay_minutes": "{:.1%}",
                                       "cumulative_share": "{:.1%}",
                                       "mean_delay_min": "{:.1f}"}),
                     use_container_width=True)
    fig("18_business_economics.png")
    md = load_report("business_insights.md")
    if md:
        with st.expander("Full written insight report"):
            st.markdown(md)
