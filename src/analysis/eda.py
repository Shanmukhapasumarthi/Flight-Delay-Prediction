from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.analysis.style import ACCENT, ACCENT_2, GOOD, INK, MUTED, SEQ, apply_style, despine
from src.config import FIGURES, PROCESSED, REPORTS

apply_style()
CMAP = LinearSegmentedColormap.from_list("delay", ["#eef3f6", "#7aa8ba", "#e3b23c", "#c8553d"])
FACTS: dict[str, float | str] = {}


def _save(fig, name: str) -> None:
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"  [eda] {name}")


# --------------------------------------------------------------------------
def fig_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    d = df["departure_delay_min"]

    axes[0].hist(d.clip(-30, 180), bins=70, color=ACCENT_2, edgecolor="white", linewidth=0.4)
    axes[0].axvline(15, color=ACCENT, ls="--", lw=1.5, label="15 min threshold")
    axes[0].set(title="Departure delay distribution", xlabel="minutes", ylabel="flights")
    axes[0].legend()

    tail = d[d > 0]
    axes[1].hist(tail.clip(0, 400), bins=60, color=ACCENT, edgecolor="white", linewidth=0.4)
    axes[1].set_yscale("log")
    axes[1].set(title="Positive-delay tail (log scale)", xlabel="minutes", ylabel="flights (log)")

    share = [(d < 15).mean(), ((d >= 15) & (d < 60)).mean(),
             ((d >= 60) & (d < 120)).mean(), (d >= 120).mean()]
    labels = ["on time\n(<15m)", "minor\n15-60m", "major\n60-120m", "severe\n>120m"]
    axes[2].bar(labels, share, color=[GOOD, "#e3b23c", "#d98d3a", ACCENT])
    for i, v in enumerate(share):
        axes[2].text(i, v + 0.008, f"{v:.1%}", ha="center", fontsize=9, color=INK)
    axes[2].set(title="Severity mix", ylabel="share of flights", ylim=(0, max(share) * 1.18))
    for a in axes:
        despine(a)
    fig.suptitle("Delay distribution is heavy-tailed: most flights are fine, a few are very late",
                 y=1.03, fontsize=11, color=MUTED)
    _save(fig, "01_delay_distribution.png")

    FACTS["pct_on_time"] = float(share[0])
    FACTS["pct_severe"] = float(share[3])
    FACTS["median_delay_when_delayed"] = float(d[d >= 15].median())
    FACTS["mean_delay_when_delayed"] = float(d[d >= 15].mean())


def fig_by_carrier(df: pd.DataFrame) -> None:
    g = (df.groupby(["carrier_code", "carrier_name"])
           .agg(rate=("is_delayed", "mean"), n=("is_delayed", "size"),
                mean_min=("departure_delay_min", "mean"),
                p90=("departure_delay_min", lambda s: s.quantile(0.9)))
           .reset_index().sort_values("rate"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = [GOOD if r < df.is_delayed.mean() else ACCENT for r in g["rate"]]
    axes[0].barh(g["carrier_code"], g["rate"], color=colors)
    axes[0].axvline(df.is_delayed.mean(), color=INK, ls="--", lw=1,
                    label=f"system {df.is_delayed.mean():.1%}")
    for i, (r, n) in enumerate(zip(g["rate"], g["n"])):
        axes[0].text(r + 0.004, i, f"{r:.1%}  (n={n/1000:.0f}k)", va="center", fontsize=8, color=MUTED)
    axes[0].set(title="Delay rate by carrier", xlabel="P(departure delay ≥ 15 min)",
                xlim=(0, g["rate"].max() * 1.35))
    axes[0].legend(loc="lower right")

    data = [df.loc[df.carrier_code == c, "departure_delay_min"].clip(-20, 150)
            for c in g["carrier_code"]]
    bp = axes[1].boxplot(data, vert=False, tick_labels=g["carrier_code"], showfliers=False,
                         patch_artist=True, medianprops=dict(color=INK))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor(c)
    axes[1].set(title="Delay minutes by carrier", xlabel="departure delay (min)")
    for a in axes:
        despine(a)
    _save(fig, "02_delay_by_carrier.png")

    FACTS["best_carrier"] = f"{g.iloc[0]['carrier_name']} ({g.iloc[0]['rate']:.1%})"
    FACTS["worst_carrier"] = f"{g.iloc[-1]['carrier_name']} ({g.iloc[-1]['rate']:.1%})"
    FACTS["carrier_spread_pp"] = float((g.iloc[-1]["rate"] - g.iloc[0]["rate"]) * 100)
    g.to_csv(REPORTS / "carrier_performance.csv", index=False)


def fig_by_airport(df: pd.DataFrame) -> None:
    g = (df.groupby("origin")
           .agg(rate=("is_delayed", "mean"), n=("is_delayed", "size"),
                total_min=("departure_delay_min", lambda s: s.clip(lower=0).sum()))
           .reset_index())
    g["share_of_delay_minutes"] = g["total_min"] / g["total_min"].sum()
    top_rate = g.sort_values("rate", ascending=False).head(15)
    top_contrib = g.sort_values("share_of_delay_minutes", ascending=False).head(15)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    axes[0].barh(top_rate["origin"][::-1], top_rate["rate"][::-1], color=ACCENT)
    axes[0].axvline(df.is_delayed.mean(), color=INK, ls="--", lw=1)
    axes[0].set(title="Worst 15 origins by delay RATE", xlabel="P(delay ≥ 15 min)")
    axes[1].barh(top_contrib["origin"][::-1], top_contrib["share_of_delay_minutes"][::-1],
                 color=ACCENT_2)
    for i, v in enumerate(top_contrib["share_of_delay_minutes"][::-1]):
        axes[1].text(v + 0.001, i, f"{v:.1%}", va="center", fontsize=8, color=MUTED)
    axes[1].set(title="Top 15 origins by SHARE of network delay minutes",
                xlabel="share of all delay minutes",
                xlim=(0, top_contrib["share_of_delay_minutes"].max() * 1.25))
    for a in axes:
        despine(a)
    fig.suptitle("Rate and impact rank differently — small airports can be late without mattering",
                 y=1.02, fontsize=11, color=MUTED)
    _save(fig, "03_delay_by_airport.png")

    t = top_contrib.iloc[0]
    FACTS["top_delay_contributor"] = f"{t['origin']} ({t['share_of_delay_minutes']:.1%} of network delay minutes)"
    FACTS["top3_airport_share"] = float(top_contrib.head(3)["share_of_delay_minutes"].sum())
    g.sort_values("share_of_delay_minutes", ascending=False).to_csv(
        REPORTS / "airport_performance.csv", index=False)


def fig_time_patterns(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    by_month = df.groupby(df["date"].dt.to_period("M"))["is_delayed"].agg(["mean", "size"])
    x = [str(p) for p in by_month.index]
    axes[0, 0].plot(x, by_month["mean"], marker="o", color=ACCENT, lw=2)
    axes[0, 0].fill_between(x, by_month["mean"], color=ACCENT, alpha=0.12)
    axes[0, 0].set(title="Delay rate by month", ylabel="P(delay ≥ 15 min)")
    axes[0, 0].tick_params(axis="x", rotation=60)

    dow = df.groupby("day_of_week")["is_delayed"].mean()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    cols = [ACCENT if v == dow.max() else (GOOD if v == dow.min() else ACCENT_2) for v in dow]
    axes[0, 1].bar(names, dow.values, color=cols)
    for i, v in enumerate(dow.values):
        axes[0, 1].text(i, v + 0.003, f"{v:.1%}", ha="center", fontsize=8, color=MUTED)
    axes[0, 1].set(title="Delay rate by day of week", ylim=(0, dow.max() * 1.2))

    hr = df.groupby("dep_hour")["is_delayed"].agg(["mean", "size"])
    ax = axes[1, 0]
    ax.bar(hr.index, hr["size"] / hr["size"].max() * hr["mean"].max(),
           color=ACCENT_2, alpha=0.22, label="relative volume")
    ax.plot(hr.index, hr["mean"], marker="o", color=ACCENT, lw=2, label="delay rate")
    ax.set(title="Delay rate by scheduled departure hour (local)",
           xlabel="hour", ylabel="P(delay ≥ 15 min)", xticks=range(0, 24, 2))
    ax.legend()

    daily = df.groupby("date")["is_delayed"].mean()
    axes[1, 1].plot(daily.index, daily.values, color="#9fb4c2", lw=0.7, label="daily")
    axes[1, 1].plot(daily.index, daily.rolling(14, center=True).mean(),
                    color=ACCENT, lw=2, label="14-day mean")
    axes[1, 1].set(title="Daily delay rate over time", ylabel="P(delay ≥ 15 min)")
    axes[1, 1].tick_params(axis="x", rotation=30)
    axes[1, 1].legend()
    for a in axes.ravel():
        despine(a)
    _save(fig, "04_time_patterns.png")

    FACTS["worst_month"] = str(by_month["mean"].idxmax())
    FACTS["worst_month_rate"] = float(by_month["mean"].max())
    FACTS["best_month"] = str(by_month["mean"].idxmin())
    FACTS["worst_dow"] = names[int(dow.idxmax())]
    FACTS["worst_dow_rate"] = float(dow.max())
    FACTS["best_dow"] = names[int(dow.idxmin())]
    FACTS["peak_hour"] = int(hr["mean"].idxmax())
    FACTS["peak_hour_rate"] = float(hr["mean"].max())
    FACTS["early_hour_rate"] = float(hr.loc[6, "mean"])


def fig_heatmaps(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    piv = df.pivot_table(index="day_of_week", columns="dep_hour",
                         values="is_delayed", aggfunc="mean")
    im = axes[0].imshow(piv.values, aspect="auto", cmap=CMAP, origin="upper")
    axes[0].set(title="Delay rate: day of week x departure hour",
                xticks=range(0, 24, 2), xlabel="local departure hour",
                yticks=range(7), yticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], label="P(delay ≥ 15 min)")

    top = df["origin"].value_counts().head(18).index
    piv2 = (df[df.origin.isin(top)]
            .pivot_table(index="origin", columns="month", values="is_delayed", aggfunc="mean")
            .loc[top])
    im2 = axes[1].imshow(piv2.values, aspect="auto", cmap=CMAP)
    axes[1].set(title="Delay rate: airport x month", xlabel="month",
                xticks=range(len(piv2.columns)), xticklabels=piv2.columns,
                yticks=range(len(piv2.index)), yticklabels=piv2.index)
    axes[1].grid(False)
    fig.colorbar(im2, ax=axes[1], label="P(delay ≥ 15 min)")
    _save(fig, "05_heatmaps.png")

    worst_cell = piv.stack().idxmax()
    FACTS["worst_dow_hour"] = (f"{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][worst_cell[0]]} "
                               f"{worst_cell[1]:02d}:00 ({piv.stack().max():.1%})")


def fig_weather(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    base = df.is_delayed.mean()

    def cat_plot(ax, col, title, order=None):
        g = df.groupby(col, observed=True)["is_delayed"].agg(["mean", "size"])
        if order is not None:
            g = g.loc[[o for o in order if o in g.index]]
        ax.bar(range(len(g)), g["mean"], color=SEQ[:len(g)])
        for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(i, m + 0.006, f"{m:.0%}", ha="center", fontsize=8, color=INK)
        ax.axhline(base, color=INK, ls="--", lw=1)
        ax.set(title=title, xticks=range(len(g)),
               xticklabels=[str(i) for i in g.index], ylim=(0, g["mean"].max() * 1.25))
        ax.tick_params(axis="x", rotation=20)

    cat_plot(axes[0, 0], "dep_rain_intensity", "Precipitation intensity at origin",
             ["none", "light", "moderate", "heavy"])
    cat_plot(axes[0, 1], "dep_wind_category", "Wind / gust category at origin",
             ["calm", "breezy", "strong", "gale"])
    cat_plot(axes[0, 2], "dep_visibility_level", "Visibility at origin",
             ["good", "moderate", "poor", "very_poor"])
    cat_plot(axes[1, 0], "dep_wx_condition", "Reported condition at origin")

    sev_bin = pd.cut(df["dep_wx_severity"], [-0.01, 0.05, 0.3, 0.7, 1.2, 2, 10],
                     labels=["none", "trace", "low", "moderate", "high", "severe"])
    g = df.groupby(sev_bin, observed=True)["is_delayed"].agg(["mean", "size"])
    axes[1, 1].plot(range(len(g)), g["mean"], marker="o", color=ACCENT, lw=2)
    axes[1, 1].set(title="Composite weather severity at origin", xticks=range(len(g)),
                   xticklabels=list(g.index), ylabel="P(delay ≥ 15 min)")
    axes[1, 1].tick_params(axis="x", rotation=20)

    cong = pd.cut(df["origin_congestion_ratio"], [0, 0.5, 0.75, 1.0, 1.25, 1.5, 10],
                  labels=["<50%", "50-75%", "75-100%", "100-125%", "125-150%", ">150%"])
    g2 = df.groupby(cong, observed=True)["is_delayed"].mean()
    axes[1, 2].plot(range(len(g2)), g2.values, marker="s", color=ACCENT_2, lw=2)
    axes[1, 2].set(title="Airport congestion (demand / declared capacity)",
                   xticks=range(len(g2)), xticklabels=list(g2.index),
                   ylabel="P(delay ≥ 15 min)")
    axes[1, 2].tick_params(axis="x", rotation=20)
    for a in axes.ravel():
        despine(a)
    _save(fig, "06_weather_and_congestion.png")

    storm = df.groupby("dep_is_storm")["is_delayed"].mean()
    FACTS["storm_delay_rate"] = float(storm.get(1, np.nan))
    FACTS["no_storm_delay_rate"] = float(storm.get(0, np.nan))
    FACTS["storm_multiplier"] = float(storm.get(1, np.nan) / storm.get(0, np.nan))
    lifr = df.groupby("dep_lifr")["is_delayed"].mean()
    FACTS["lifr_delay_rate"] = float(lifr.get(1, np.nan))
    bad = df["dep_wx_severity"] > 0.3
    FACTS["share_flights_adverse_wx"] = float(bad.mean())
    excess = (df.loc[bad, "is_delayed"].mean() - df.loc[~bad, "is_delayed"].mean())
    FACTS["excess_delays_from_wx"] = float(excess * bad.mean() / df.is_delayed.mean())


def fig_geo(df: pd.DataFrame) -> None:
    ap = (df.groupby(["origin", "origin_latitude", "origin_longitude"])
            .agg(rate=("is_delayed", "mean"), n=("is_delayed", "size"))
            .reset_index())
    routes = (df.groupby(["origin", "destination"])
                .agg(n=("is_delayed", "size"), rate=("is_delayed", "mean"))
                .reset_index().sort_values("n", ascending=False).head(70))
    pos = ap.set_index("origin")[["origin_longitude", "origin_latitude"]].to_dict("index")

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    for _, r in routes.iterrows():
        if r["origin"] in pos and r["destination"] in pos:
            o, d = pos[r["origin"]], pos[r["destination"]]
            ax.plot([o["origin_longitude"], d["origin_longitude"]],
                    [o["origin_latitude"], d["origin_latitude"]],
                    color=MUTED, alpha=0.16, lw=0.6 + r["n"] / routes["n"].max() * 1.6, zorder=1)
    sc = ax.scatter(ap["origin_longitude"], ap["origin_latitude"],
                    s=ap["n"] / ap["n"].max() * 620 + 25, c=ap["rate"], cmap=CMAP,
                    edgecolor="white", linewidth=1.1, zorder=3, vmin=ap["rate"].min())
    for _, r in ap.iterrows():
        ax.annotate(r["origin"], (r["origin_longitude"], r["origin_latitude"]),
                    fontsize=7.5, color=INK, xytext=(0, 9), textcoords="offset points",
                    ha="center", zorder=4)
    fig.colorbar(sc, ax=ax, label="P(departure delay ≥ 15 min)", shrink=0.8)
    ax.set(title="Network geography: bubble size = departures, colour = delay rate",
           xlabel="longitude", ylabel="latitude")
    despine(ax)
    _save(fig, "07_geographic_map.png")


def fig_propagation_and_holidays(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ub = pd.cut(df["upstream_delay_min"], [-0.01, 0.1, 15, 30, 60, 120, 10000],
                labels=["0", "1-15", "15-30", "30-60", "60-120", ">120"])
    g = df.groupby(ub, observed=True)["is_delayed"].agg(["mean", "size"])
    axes[0].bar(range(len(g)), g["mean"], color=SEQ[:len(g)])
    for i, m in enumerate(g["mean"]):
        axes[0].text(i, m + 0.01, f"{m:.0%}", ha="center", fontsize=8, color=INK)
    axes[0].set(title="Delay propagation from previous leg", xticks=range(len(g)),
                xticklabels=list(g.index), xlabel="upstream delay (min)",
                ylabel="P(delay ≥ 15 min)", ylim=(0, g["mean"].max() * 1.2))

    leg = df.groupby("leg_depth")["is_delayed"].agg(["mean", "size"])
    leg = leg[leg["size"] > 500]
    axes[1].plot(leg.index, leg["mean"], marker="o", color=ACCENT, lw=2)
    axes[1].set(title="Delay rate by position in the daily rotation",
                xlabel="leg number (0 = first flight of the day)",
                ylabel="P(delay ≥ 15 min)")

    hol = df.groupby("days_from_major_holiday")["is_delayed"].agg(["mean", "size"])
    hol = hol[(hol.index >= -7) & (hol.index <= 7)]
    cols = [ACCENT if abs(i) <= 2 else ACCENT_2 for i in hol.index]
    axes[2].bar(hol.index, hol["mean"], color=cols)
    axes[2].axhline(df.is_delayed.mean(), color=INK, ls="--", lw=1, label="system mean")
    axes[2].set(title="Days relative to a major travel holiday",
                xlabel="days from holiday (0 = holiday)", ylabel="P(delay ≥ 15 min)")
    axes[2].legend()
    for a in axes:
        despine(a)
    _save(fig, "08_propagation_and_holidays.png")

    FACTS["prop_no_upstream"] = float(g.loc["0", "mean"])
    FACTS["prop_gt60"] = float(g.loc["60-120", "mean"])
    FACTS["first_leg_rate"] = float(leg["mean"].iloc[0])
    FACTS["last_leg_rate"] = float(leg["mean"].iloc[-1])
    hw = df[df["in_holiday_window"] == 1]["is_delayed"].mean()
    nw = df[df["in_holiday_window"] == 0]["is_delayed"].mean()
    FACTS["holiday_window_rate"] = float(hw)
    FACTS["holiday_uplift_pct"] = float((hw / nw - 1) * 100)


def fig_correlation(df: pd.DataFrame) -> None:
    cols = ["is_delayed", "departure_delay_min", "dep_wx_severity", "arr_wx_severity",
            "dep_wx_precip_mm", "dep_wx_wind_gust_kt", "dep_wx_visibility_km",
            "origin_congestion_ratio", "bank_pressure_3h", "dep_time_decimal",
            "upstream_delay_min", "slack_vs_upstream_min", "leg_depth",
            "carrier_delay_rate_7d", "route_delay_rate_30d", "origin_delay_rate_7d",
            "distance_km", "schedule_padding_min", "aircraft_age_years",
            "holiday_proximity", "seats"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=range(len(cols)), yticks=range(len(cols)),
           xticklabels=cols, yticklabels=cols, title="Correlation matrix (numeric features)")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            if abs(v) > 0.28 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.75)
    _save(fig, "09_correlation_matrix.png")

    tc = corr["is_delayed"].drop(["is_delayed", "departure_delay_min"]).abs().sort_values(ascending=False)
    FACTS["top_linear_correlate"] = f"{tc.index[0]} (r={corr['is_delayed'][tc.index[0]]:.2f})"


# --------------------------------------------------------------------------
def write_summary(df: pd.DataFrame) -> None:
    f = FACTS
    md = f"""# Exploratory Data Analysis — Summary

Window **{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}**,
**{len(df):,}** operated flights, **{df['origin'].nunique()}** airports,
**{df['carrier_code'].nunique()}** carriers, **{df['route'].nunique():,}** routes.
System delay rate (≥15 min): **{df['is_delayed'].mean():.1%}**.

## Shape of the problem
- {f['pct_on_time']:.1%} of flights depart within 15 minutes of schedule;
  {f['pct_severe']:.1%} are more than two hours late.
- When a flight is delayed, the median delay is {f['median_delay_when_delayed']:.0f} min
  and the mean is {f['mean_delay_when_delayed']:.0f} min — the distribution is heavy-tailed,
  so averages understate the operational pain.

## Which airlines
- Best: **{f['best_carrier']}**; worst: **{f['worst_carrier']}**.
- The spread between best and worst carrier is **{f['carrier_spread_pp']:.1f} percentage points**.
  Regional carriers are systematically worse, consistent with short turnarounds and
  deeper daily rotations.

## Which airports
- Largest single contributor to network delay minutes: **{f['top_delay_contributor']}**.
- The top 3 origins account for **{f['top3_airport_share']:.1%}** of all delay minutes.
- Delay *rate* and delay *impact* rank differently: a small station can be chronically
  late without mattering to the network.

## When
- Worst month: **{f['worst_month']}** ({f['worst_month_rate']:.1%}); best: **{f['best_month']}**.
- Worst weekday: **{f['worst_dow']}** ({f['worst_dow_rate']:.1%}); best: **{f['best_dow']}**.
- Delay risk compounds through the day: {f['early_hour_rate']:.1%} at 06:00 vs
  **{f['peak_hour_rate']:.1%} at {f['peak_hour']:02d}:00**.
- Worst single slot in the week: **{f['worst_dow_hour']}**.

## Weather
- Thunderstorms at the origin lift the delay rate to **{f['storm_delay_rate']:.1%}**
  vs {f['no_storm_delay_rate']:.1%} otherwise — a **{f['storm_multiplier']:.1f}x** multiplier.
- Low-IFR conditions (ceiling < 500 ft or visibility < 1.6 km): {f['lifr_delay_rate']:.1%}.
- {f['share_flights_adverse_wx']:.1%} of flights face non-trivial adverse weather; they
  account for an estimated **{f['excess_delays_from_wx']:.1%}** of excess delays.

## Congestion and propagation
- Delay rate rises monotonically with the demand/capacity ratio at the origin.
- Propagation is the single strongest operational lever: with no upstream delay the rate
  is {f['prop_no_upstream']:.1%}; with 60–120 minutes of inherited delay it is
  **{f['prop_gt60']:.1%}**.
- First leg of the day: {f['first_leg_rate']:.1%} → deepest leg: **{f['last_leg_rate']:.1%}**.

## Holidays
- Flights inside a ±3-day major-holiday window run at {f['holiday_window_rate']:.1%},
  **{f['holiday_uplift_pct']:+.0f}%** relative to normal days.

## Note on linear correlation
- Strongest single linear correlate of the target: {f['top_linear_correlate']}.
  All individual correlations are modest, which is exactly why a non-linear model with
  interaction terms is warranted.
"""
    (REPORTS / "eda_summary.md").write_text(md, encoding="utf-8")
    (REPORTS / "eda_facts.json").write_text(json.dumps(f, indent=2), encoding="utf-8")


def main() -> None:
    df = pd.read_parquet(PROCESSED / "features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    print(f"[eda] {len(df):,} flights")
    fig_distribution(df)
    fig_by_carrier(df)
    fig_by_airport(df)
    fig_time_patterns(df)
    fig_heatmaps(df)
    fig_weather(df)
    fig_geo(df)
    fig_propagation_and_holidays(df)
    fig_correlation(df)
    write_summary(df)
    print(f"[eda] figures -> {FIGURES}, summary -> {REPORTS/'eda_summary.md'}")


if __name__ == "__main__":
    main()
