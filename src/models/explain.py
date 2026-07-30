"""Stage 9 - Model Explainability (SHAP).

Two audiences, two outputs:

  GLOBAL  which factors drive delay risk across the network, aggregated by
          feature family, plus dependence plots for the strongest drivers

  LOCAL   "why is flight AA1423 flagged as high risk?" — a per-flight
          attribution translated into operational language, which is what
          the API returns and the dashboard shows

The explainer runs on the uncalibrated tree model. Isotonic calibration is a
monotone transform of the score, so it changes the probability but not the
ordering or the sign of any contribution.
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.analysis.style import ACCENT, ACCENT_2, GOOD, INK, MUTED, apply_style, despine
from src.config import FIGURES, MODELS_DIR, REPORTS, SEED
from src.features.build_features import FEATURE_GROUPS
from src.models.dataset import make_splits

warnings.filterwarnings("ignore")
apply_style()

GROUP_OF = {f: g for g, cols in FEATURE_GROUPS.items() for f in cols}

# feature -> (label, unit, direction phrasing)
REASONS = {
    "upstream_delay_min": ("Inbound aircraft late", "min"),
    "slack_vs_upstream_min": ("Ground-time slack", "min"),
    "aircraft_cum_delay_today": ("Delay accumulated by this airframe today", "min"),
    "prev_leg_known": ("Inbound leg status confirmed", ""),
    "scheduled_turnaround_min": ("Scheduled turnaround", "min"),
    "origin_congestion_ratio": ("Departure-bank load vs. capacity", "x"),
    "carrier_delay_rate_7d": ("Carrier 7-day delay rate", "%"),
    "origin_delay_rate_7d": ("Origin 7-day delay rate", "%"),
    "route_delay_rate_30d": ("Route 30-day delay rate", "%"),
    "carrier_origin_delay_rate_14d": ("Carrier-at-station 14-day delay rate", "%"),
    "origin_mean_delay_min_7d": ("Origin mean delay, last 7 days", "min"),
    "dep_wx_severity": ("Weather severity at origin", "index"),
    "route_wx_severity": ("Weather severity across the route", "index"),
    "dep_wx_visibility_km": ("Visibility at origin", "km"),
    "dep_wx_wind_speed_kt": ("Wind at origin", "kt"),
    "dep_gust_factor": ("Gust factor at origin", "x"),
    "dep_wx_cloud_ceiling_ft": ("Cloud ceiling at origin", "ft"),
    "dep_wx_temperature_c": ("Temperature at origin", "°C"),
    "dep_temp_anomaly": ("Temperature vs. seasonal normal", "°C"),
    "dep_dewpoint_spread": ("Dewpoint spread (fog proxy)", "°C"),
    "dep_time_decimal": ("Departure time of day", "h"),
    "hour_sin": ("Time-of-day position", ""),
    "hour_cos": ("Time-of-day position", ""),
    "doy_sin": ("Season", ""),
    "doy_cos": ("Season", ""),
    "day_of_week": ("Day of week", ""),
    "dep_minute": ("Minutes past the hour", "min"),
    "time_block": ("Departure period", ""),
    "holiday_proximity": ("Proximity to a major holiday", ""),
    "schedule_padding_min": ("Schedule padding", "min"),
    "padding_ratio": ("Schedule padding share", ""),
    "scheduled_block_min": ("Scheduled block time", "min"),
    "aircraft_age_years": ("Airframe age", "yr"),
    "seats": ("Aircraft size", "seats"),
    "aircraft_model": ("Aircraft type", ""),
    "carrier_code": ("Operating carrier", ""),
    "origin": ("Origin station", ""),
    "destination": ("Destination station", ""),
    "carrier_share_at_origin": ("Carrier share of the station", ""),
    "carrier_volume_7d": ("Carrier weekly volume", "flights"),
    "carrier_origin_volume_14d": ("Carrier-at-station volume", "flights"),
    "route_popularity_30d": ("Route frequency", "flights/30d"),
    "lat_delta": ("North-south routing", "°"),
    "bearing_sin": ("Route direction", ""),
    "bearing_cos": ("Route direction", ""),
}


def label_for(feature: str) -> str:
    return REASONS.get(feature, (feature.replace("_", " ").capitalize(), ""))[0]


def format_value(feature: str, value) -> str:
    unit = REASONS.get(feature, ("", ""))[1]
    if isinstance(value, str) or value is None or (isinstance(value, float) and np.isnan(value)):
        return str(value)
    if unit == "%":
        return f"{value*100:.0f}%"
    if unit in ("min", "ft", "flights", "flights/30d", "seats"):
        return f"{value:,.0f} {unit}".strip()
    if unit == "h":
        return f"{int(value):02d}:{int(round((value % 1) * 60)):02d}"
    return f"{value:,.1f} {unit}".strip()


# --------------------------------------------------------------------------
def load_bundle(path=None):
    return joblib.load(path or MODELS_DIR / "final_model.joblib")


def get_explainer(bundle):
    return shap.TreeExplainer(bundle["model"])


def explain_rows(bundle, X: pd.DataFrame, top_n: int = 5,
                 explainer=None) -> list[list[dict]]:
    """Per-row attribution, ordered by contribution to delay risk."""
    explainer = explainer or get_explainer(bundle)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    out = []
    for i in range(len(X)):
        contrib = sorted(zip(X.columns, sv[i], X.iloc[i].tolist()),
                         key=lambda t: -abs(t[1]))[:top_n]
        out.append([{
            "feature": f,
            "label": label_for(f),
            "value": (None if (isinstance(v, float) and np.isnan(v)) else
                      (str(v) if isinstance(v, str) else float(v))),
            "display_value": format_value(f, v),
            "shap_value": float(s),
            "direction": "increases risk" if s > 0 else "reduces risk",
        } for f, s, v in contrib])
    return out


def narrate(contributions: list[dict], probability: float) -> str:
    up = [c for c in contributions if c["shap_value"] > 0][:3]
    down = [c for c in contributions if c["shap_value"] < 0][:2]
    parts = [f"Predicted departure-delay risk {probability:.0%}."]
    if up:
        parts.append("Driven up by: " + "; ".join(
            f"{c['label'].lower()} ({c['display_value']})" for c in up) + ".")
    if down:
        parts.append("Offset by: " + "; ".join(
            f"{c['label'].lower()} ({c['display_value']})" for c in down) + ".")
    return " ".join(parts)


# --------------------------------------------------------------------------
def main() -> None:
    bundle = load_bundle()
    sp = make_splits(features=bundle["features"])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(sp.X_test), size=min(6000, len(sp.X_test)), replace=False)
    X = sp.X_test.iloc[idx]
    y = sp.y_test.iloc[idx]
    meta = sp.meta_test.iloc[idx]

    print(f"[shap] explaining {len(X):,} test flights")
    explainer = get_explainer(bundle)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)

    # ---- global: bar + family aggregation --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    top = mean_abs.head(20).iloc[::-1]
    palette = {"time": "#2f6f8f", "weather": "#4d8ba6", "airport": "#e3b23c",
               "flight": "#d98d3a", "geospatial": "#7aa8ba", "historical": ACCENT}
    axes[0].barh([label_for(f) for f in top.index], top.values,
                 color=[palette.get(GROUP_OF.get(f, ""), MUTED) for f in top.index])
    axes[0].set(title="Global feature impact (mean |SHAP|)", xlabel="mean |SHAP| (log-odds)")

    fam = (pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
             .groupby([GROUP_OF.get(f, "other") for f in X.columns]).sum()
             .sort_values())
    fam_pct = fam / fam.sum()
    axes[1].barh(fam_pct.index, fam_pct.values,
                 color=[palette.get(g, MUTED) for g in fam_pct.index])
    for i, v in enumerate(fam_pct.values):
        axes[1].text(v + 0.004, i, f"{v:.1%}", va="center", fontsize=9, color=INK)
    axes[1].set(title="Share of total explanatory power by feature family",
                xlabel="share of summed |SHAP|", xlim=(0, fam_pct.max() * 1.2))
    for a in axes:
        despine(a)
    fig.savefig(FIGURES / "14_shap_global.png")
    plt.close(fig)

    # ---- beeswarm --------------------------------------------------------
    try:
        expl = shap.Explanation(values=sv[:3000],
                                data=X.iloc[:3000].astype(float, errors="ignore"),
                                feature_names=[label_for(c) for c in X.columns])
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(expl, max_display=18, show=False)
        plt.title("SHAP value distribution (test window)", fontsize=12, color=INK)
        plt.tight_layout()
        plt.savefig(FIGURES / "15_shap_beeswarm.png", dpi=130)
        plt.close()
    except Exception as e:  # beeswarm is fussy with categoricals
        print(f"[shap] beeswarm skipped: {e}")

    # ---- dependence for the top numeric drivers --------------------------
    numeric_top = [f for f in mean_abs.index if X[f].dtype.kind == "f"][:3]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for ax, f in zip(axes, numeric_top):
        j = list(X.columns).index(f)
        v = X[f].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(v, [1, 99])
        m = (v >= lo) & (v <= hi)
        ax.scatter(v[m], sv[m, j], s=5, alpha=0.25, color=ACCENT_2)
        b = pd.DataFrame({"v": v[m], "s": sv[m, j]})
        b["bin"] = pd.qcut(b["v"], 18, duplicates="drop")
        g = b.groupby("bin", observed=True).agg(v=("v", "mean"), s=("s", "mean"))
        ax.plot(g["v"], g["s"], color=ACCENT, lw=2)
        ax.axhline(0, color=MUTED, lw=1, ls="--")
        ax.set(title=label_for(f), xlabel=f, ylabel="SHAP (log-odds)")
        despine(ax)
    fig.suptitle("How the model responds to its strongest drivers", y=1.02,
                 fontsize=11, color=MUTED)
    fig.savefig(FIGURES / "16_shap_dependence.png")
    plt.close(fig)

    # ---- local: a real high-risk flight ----------------------------------
    proba = bundle["calibrator"].predict_proba(X)[:, 1]
    order = np.argsort(proba)[::-1]
    picks = {"highest_risk": int(order[0]),
             "typical": int(order[len(order) // 2]),
             "lowest_risk": int(order[-1])}

    examples = {}
    for tag, i in picks.items():
        contrib = explain_rows(bundle, X.iloc[[i]], top_n=6, explainer=explainer)[0]
        m = meta.iloc[i]
        examples[tag] = {
            "flight": f"{m['carrier_code']}{int(m['flight_number'])}",
            "route": f"{m['origin']}-{m['destination']}",
            "scheduled_departure_local": str(m["scheduled_departure_local"]),
            "predicted_probability": float(proba[i]),
            "actual_departure_delay_min": float(m["departure_delay_min"]),
            "actually_delayed": int(m["is_delayed"]),
            "contributions": contrib,
            "narrative": narrate(contrib, float(proba[i])),
        }

    ex = examples["highest_risk"]
    fig, ax = plt.subplots(figsize=(10, 5))
    c = ex["contributions"][::-1]
    vals = [x["shap_value"] for x in c]
    ax.barh([f"{x['label']}\n({x['display_value']})" for x in c], vals,
            color=[ACCENT if v > 0 else GOOD for v in vals])
    ax.axvline(0, color=INK, lw=1)
    ax.set(title=f"Why {ex['flight']} ({ex['route']}) is flagged: "
                 f"{ex['predicted_probability']:.0%} risk "
                 f"(actual delay {ex['actual_departure_delay_min']:.0f} min)",
           xlabel="contribution to delay log-odds")
    despine(ax)
    fig.savefig(FIGURES / "17_shap_local_example.png")
    plt.close(fig)

    (REPORTS / "shap_examples.json").write_text(
        json.dumps(examples, indent=2, default=float), encoding="utf-8")
    fam_pct.sort_values(ascending=False).to_csv(REPORTS / "shap_family_share.csv")
    mean_abs.to_csv(REPORTS / "shap_global_importance.csv")

    print("[shap] family shares:", {k: f"{v:.1%}" for k, v in
                                    fam_pct.sort_values(ascending=False).items()})
    print("[shap] example:", examples["highest_risk"]["narrative"])


if __name__ == "__main__":
    main()
