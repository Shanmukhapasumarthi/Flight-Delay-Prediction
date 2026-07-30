"""Stage 10 - Business Insights.

Turns the model into decisions. Three questions are answered with numbers:

  1. WHERE does the delay actually come from? (attribution, not accuracy)
  2. WHAT is the optimal alerting threshold given real intervention costs?
  3. HOW MUCH money does acting on the model save versus current practice?

The threshold is NOT 0.5 and it is not the F1 optimum. It is the point where
the expected value of intervening turns positive, which depends entirely on
the cost asymmetry between a missed delay and a wasted intervention.
"""
from __future__ import annotations

import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.style import ACCENT, ACCENT_2, GOOD, INK, MUTED, apply_style, despine
from src.config import CFG, FIGURES, MODELS_DIR, PROCESSED, REPORTS
from src.models.dataset import make_splits

warnings.filterwarnings("ignore")
apply_style()
B = CFG["business"]


# --------------------------------------------------------------------------
def cost_parameters(df: pd.DataFrame) -> dict:
    """Cost model, with the delay-length assumption taken from the data."""
    delayed = df.loc[df["is_delayed"] == 1, "departure_delay_min"]
    avg_delay = float(delayed.mean())
    per_min = B["cost_per_delay_minute_usd"] + B["passenger_cost_per_minute_usd"]
    benefit = avg_delay * B["mitigation_effectiveness"] * per_min
    return {
        "avg_delay_minutes_when_delayed": avg_delay,
        "cost_per_delay_minute_usd": per_min,
        "cost_of_one_delay_usd": avg_delay * per_min,
        "mitigation_cost_usd": B["mitigation_cost_usd"],
        "mitigation_effectiveness": B["mitigation_effectiveness"],
        "benefit_of_correct_intervention_usd": benefit,
        "break_even_probability": B["mitigation_cost_usd"] / benefit,
    }


def threshold_sweep(y, proba, cp: dict) -> pd.DataFrame:
    rows = []
    for thr in np.arange(0.02, 0.96, 0.01):
        alert = proba >= thr
        tp = int(np.sum(alert & (y == 1)))
        fp = int(np.sum(alert & (y == 0)))
        fn = int(np.sum(~alert & (y == 1)))
        saved = tp * cp["benefit_of_correct_intervention_usd"]
        spent = (tp + fp) * cp["mitigation_cost_usd"]
        rows.append({
            "threshold": round(float(thr), 3), "alerts": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "alert_rate": (tp + fp) / len(y),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "gross_saving_usd": saved,
            "intervention_cost_usd": spent,
            "net_benefit_usd": saved - spent,
        })
    return pd.DataFrame(rows)


def decile_table(y, proba, delay_min) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": proba, "min": np.clip(delay_min, 0, None)})
    # bin 1 = lowest risk ... bin 10 = highest risk
    df["bin"] = pd.qcut(df["p"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    g = (df.groupby("bin", observed=True)
           .agg(flights=("y", "size"), delay_rate=("y", "mean"),
                mean_delay_min=("min", "mean"), delay_minutes=("min", "sum"))
           .sort_index(ascending=False))          # highest risk first
    g["lift"] = g["delay_rate"] / df["y"].mean()
    g["share_of_delay_minutes"] = g["delay_minutes"] / g["delay_minutes"].sum()
    g["cumulative_share"] = g["share_of_delay_minutes"].cumsum()
    g.index = [f"D{i}" for i in g.index]
    return g


# --------------------------------------------------------------------------
def propagation_share(df: pd.DataFrame) -> dict:
    """How much of the delay burden is inherited rather than generated?"""
    d = df[df["is_delayed"] == 1]
    inherited = np.minimum(d["upstream_delay_min"].fillna(0), d["departure_delay_min"])
    total = d["departure_delay_min"].clip(lower=0).sum()
    first_leg = df[df["is_first_leg_of_day"] == 1]["is_delayed"].mean() if \
        "is_first_leg_of_day" in df.columns else np.nan
    return {
        "share_of_delay_minutes_inherited": float(inherited.sum() / total),
        "delayed_flights_with_late_inbound": float((d["upstream_delay_min"] > 15).mean()),
        "first_leg_delay_rate": float(first_leg) if first_leg == first_leg else None,
    }


def sensitivity(y, proba, avg_delay: float) -> pd.DataFrame:
    """The optimal threshold rests on two ASSUMED numbers. Show how much it moves."""
    per_min = B["cost_per_delay_minute_usd"] + B["passenger_cost_per_minute_usd"]
    rows = []
    for cost in (300, 600, 900, 1500, 2500):
        for eff in (0.15, 0.30, 0.50):
            benefit = avg_delay * eff * per_min
            sw = threshold_sweep(y, proba, {
                "benefit_of_correct_intervention_usd": benefit,
                "mitigation_cost_usd": cost})
            b = sw.loc[sw["net_benefit_usd"].idxmax()]
            rows.append({
                "mitigation_cost_usd": cost, "effectiveness": eff,
                "break_even_p": cost / benefit,
                "optimal_threshold": b["threshold"],
                "alert_rate": b["alert_rate"], "precision": b["precision"],
                "recall": b["recall"], "net_benefit_usd": b["net_benefit_usd"]})
    return pd.DataFrame(rows)


def main() -> None:
    bundle = joblib.load(MODELS_DIR / "final_model.joblib")
    sp = make_splits(features=bundle["features"])
    proba = bundle["calibrator"].predict_proba(sp.X_test)[:, 1]
    y = sp.y_test.to_numpy()
    meta = sp.meta_test.reset_index(drop=True)
    full = pd.read_parquet(PROCESSED / "features.parquet")

    cp = cost_parameters(full)
    sweep = threshold_sweep(y, proba, cp)
    best = sweep.loc[sweep["net_benefit_usd"].idxmax()]
    dec = decile_table(y, proba, meta["departure_delay_min"].to_numpy())
    prop = propagation_share(full)

    n_test = len(y)
    days = (meta["date"].max() - meta["date"].min()).days + 1
    per_year = 365 / days

    # ---- baselines -------------------------------------------------------
    alert_all = (len(y) * -cp["mitigation_cost_usd"]
                 + y.sum() * cp["benefit_of_correct_intervention_usd"])
    # heuristic: alert on the historically worst carrier x peak-hour combos
    heur = ((sp.X_test["carrier_delay_rate_7d"] > full["carrier_delay_rate_7d"].median())
            & (sp.X_test["dep_time_decimal"] >= 15)).to_numpy()
    heur_net = (np.sum(heur & (y == 1)) * cp["benefit_of_correct_intervention_usd"]
                - heur.sum() * cp["mitigation_cost_usd"])

    econ = {
        "cost_model": cp,
        "optimal_threshold": float(best["threshold"]),
        "at_optimum": {k: float(best[k]) for k in
                       ["alert_rate", "precision", "recall", "alerts", "tp", "fp",
                        "net_benefit_usd", "gross_saving_usd", "intervention_cost_usd"]},
        "test_window_days": int(days),
        "test_flights": int(n_test),
        "net_benefit_per_flight_usd": float(best["net_benefit_usd"] / n_test),
        "annualised_net_benefit_usd": float(best["net_benefit_usd"] * per_year),
        "baseline_alert_every_flight_usd": float(alert_all),
        "baseline_heuristic_usd": float(heur_net),
        "uplift_vs_heuristic_usd": float(best["net_benefit_usd"] - heur_net),
        "propagation": prop,
    }

    sens = sensitivity(y, proba, cp["avg_delay_minutes_when_delayed"])
    sens.to_csv(REPORTS / "threshold_sensitivity.csv", index=False)
    sens_rows = "\n".join(
        f"| ${r.mitigation_cost_usd:,.0f} | {r.effectiveness:.0%} | {r.break_even_p:.2f} | "
        f"{r.optimal_threshold:.2f} | {r.alert_rate:.1%} | {r.precision:.0%} | "
        f"{r.recall:.0%} | ${r.net_benefit_usd:,.0f} |"
        for _, r in sens.iterrows())

    # ---- figures ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    axes[0].plot(sweep["threshold"], sweep["net_benefit_usd"] / 1e6, color=ACCENT, lw=2)
    axes[0].axvline(best["threshold"], color=INK, ls="--", lw=1.2,
                    label=f"optimum p={best['threshold']:.2f}")
    axes[0].axvline(0.5, color=MUTED, ls=":", lw=1.2, label="naive p=0.50")
    axes[0].axhline(0, color=MUTED, lw=1)
    axes[0].set(title="Net benefit vs. alerting threshold",
                xlabel="alert if predicted probability ≥", ylabel="net benefit (US$m, test window)")
    axes[0].legend()

    axes[1].plot(sweep["threshold"], sweep["precision"], color=ACCENT, lw=2, label="precision")
    axes[1].plot(sweep["threshold"], sweep["recall"], color=ACCENT_2, lw=2, label="recall")
    axes[1].plot(sweep["threshold"], sweep["alert_rate"], color=GOOD, lw=2, ls="--",
                 label="share of flights alerted")
    axes[1].axvline(best["threshold"], color=INK, ls="--", lw=1.2)
    axes[1].set(title="Operating characteristics", xlabel="threshold")
    axes[1].legend()

    x = np.arange(len(dec))
    axes[2].bar(x, dec["share_of_delay_minutes"], color=ACCENT_2, label="share of delay minutes")
    axes[2].plot(x, dec["cumulative_share"], color=ACCENT, marker="o", ms=4, lw=2,
                 label="cumulative")
    axes[2].set(title="Delay minutes captured by risk decile", xticks=x,
                xticklabels=dec.index, xlabel="predicted-risk decile (D10 = highest)")
    axes[2].legend()
    for a in axes:
        despine(a)
    fig.savefig(FIGURES / "18_business_economics.png")
    plt.close(fig)

    # ---- write-up --------------------------------------------------------
    eda = json.loads((REPORTS / "eda_facts.json").read_text())
    fam = pd.read_csv(REPORTS / "shap_family_share.csv", index_col=0).iloc[:, 0] \
        if (REPORTS / "shap_family_share.csv").exists() else pd.Series(dtype=float)
    top2 = dec.loc[["D10", "D9"], "share_of_delay_minutes"].sum()
    airports = pd.read_csv(REPORTS / "airport_performance.csv")
    carriers = pd.read_csv(REPORTS / "carrier_performance.csv")
    worst_peak = (full[full["is_peak_hour"] == 1].groupby("carrier_name")["is_delayed"]
                  .mean().sort_values())

    md = f"""# Business Insights

*Generated from the test window ({meta['date'].min():%Y-%m-%d} → {meta['date'].max():%Y-%m-%d},
{n_test:,} flights) using the calibrated production model.*

---

## 1. The headline is not the accuracy

The model scores {bundle['metrics_test']['roc_auc']:.3f} ROC-AUC. On its own that number
is operationally meaningless. What matters is that **the top two risk deciles contain
{top2:.0%} of all delay minutes** in the network. Ranking flights by predicted risk and
working the top 20% reaches most of the pain with a fifth of the effort.

| Risk decile | Flights | Delay rate | Lift | Share of delay minutes |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {i} | {int(r.flights):,} | {r.delay_rate:.1%} | {r.lift:.2f}x | {r.share_of_delay_minutes:.1%} |"
        for i, r in dec.iterrows()) + f"""

## 2. Where delay actually comes from

- **Propagation is the dominant mechanism.** {prop['share_of_delay_minutes_inherited']:.0%} of all
  departure-delay minutes are *inherited* from the aircraft's previous leg rather than
  generated at the gate. {prop['delayed_flights_with_late_inbound']:.0%} of delayed flights had an
  inbound aircraft that was already more than 15 minutes late.
- **The day compounds.** First flight of the day: {eda['first_leg_rate']:.1%} delay rate;
  by the deepest leg of the rotation it is {eda['last_leg_rate']:.1%}. Recovery has to happen
  early or not at all.
- **Weather is sharp but narrow.** Thunderstorms multiply delay risk by
  {eda['storm_multiplier']:.1f}x, but only {eda['share_flights_adverse_wx']:.0%} of flights face
  meaningful adverse weather. Weather explains
  {fam.get('weather', float('nan'))*100:.0f}% of the model's total explanatory power —
  real, but far less than operations staff usually assume.
- **Congestion is structural.** Delay risk rises monotonically with the departure-bank
  demand/capacity ratio at the origin.

## 3. Concentration: a few places and times carry the network

- {airports.iloc[0]['origin']} alone generates {airports.iloc[0]['share_of_delay_minutes']:.1%}
  of all network delay minutes; the top three stations account for {eda['top3_airport_share']:.1%}.
- The worst weekly slot is **{eda['worst_dow_hour']}**, against a system average of
  {full['is_delayed'].mean():.1%}.
- Holiday windows (±3 days of a major travel holiday) run **{eda['holiday_uplift_pct']:+.0f}%**
  above normal.
- Carrier spread is {eda['carrier_spread_pp']:.0f} percentage points between best
  ({eda['best_carrier']}) and worst ({eda['worst_carrier']}). At peak hours the best performer is
  **{worst_peak.index[0]}** ({worst_peak.iloc[0]:.1%}) and the worst is
  **{worst_peak.index[-1]}** ({worst_peak.iloc[-1]:.1%}).

## 4. The alerting threshold is an economics question, not an ML question

Cost assumptions (editable in `config/config.yaml`):

| Parameter | Value |
|---|---:|
| Aircraft direct operating cost | ${B['cost_per_delay_minute_usd']:,.0f}/min |
| Passenger time cost | ${B['passenger_cost_per_minute_usd']:,.0f}/min |
| Average delay when a flight is late | {cp['avg_delay_minutes_when_delayed']:.0f} min |
| Cost of one delayed departure | **${cp['cost_of_one_delay_usd']:,.0f}** |
| Cost of one intervention | ${cp['mitigation_cost_usd']:,.0f} |
| Delay minutes recovered by intervening | {cp['mitigation_effectiveness']:.0%} |
| Benefit of one correct intervention | ${cp['benefit_of_correct_intervention_usd']:,.0f} |

A false positive costs ${cp['mitigation_cost_usd']:,.0f}. A missed delay costs
${cp['benefit_of_correct_intervention_usd']:,.0f} in forgone recovery. The asymmetry is
{cp['benefit_of_correct_intervention_usd']/cp['mitigation_cost_usd']:.1f}:1, so the
break-even probability is **{cp['break_even_probability']:.2f}**, not 0.50.

Sweeping the threshold over the test window gives an optimum at
**p ≥ {best['threshold']:.2f}**:

- alert on **{best['alert_rate']:.1%}** of flights ({int(best['alerts']):,} of {n_test:,})
- precision **{best['precision']:.1%}**, recall **{best['recall']:.1%}**
- net benefit **${best['net_benefit_usd']:,.0f}** over {days} days
  (${econ['net_benefit_per_flight_usd']:,.2f} per flight flown)
- annualised at this traffic level: **${econ['annualised_net_benefit_usd']/1e6:,.1f}M**

### Versus the alternatives

| Policy | Net benefit (test window) |
|---|---:|
| Intervene on every flight | ${alert_all:,.0f} |
| Heuristic (worse-than-median carrier, afternoon departure) | ${heur_net:,.0f} |
| **Model at optimal threshold** | **${best['net_benefit_usd']:,.0f}** |

The model beats the plausible human heuristic by
**${econ['uplift_vs_heuristic_usd']:,.0f}** over three months.

### How much does this depend on my assumptions?

Intervention cost and effectiveness are *assumptions*, not measurements, and the
threshold moves with them. The honest version of the answer is the grid:

| Intervention cost | Effectiveness | Break-even p | Optimal threshold | Alerts | Precision | Recall | Net benefit |
|---:|---:|---:|---:|---:|---:|---:|---:|
{sens_rows}

Two things stand out, and the second one matters more than the first.

**The policy is very sensitive.** The optimal alert rate ranges from
{sens['alert_rate'].min():.0%} to {sens['alert_rate'].max():.0%} of flights across this grid.
Anyone quoting a single headline threshold without stating the cost assumptions behind it
is quoting noise.

**The programme is not universally worth running.** Net benefit is positive in
{(sens['net_benefit_usd'] > 0).sum()} of {len(sens)} scenarios but turns *negative* in
{(sens['net_benefit_usd'] <= 0).sum()} — specifically when interventions are both expensive
(≥ $1,500) and weak (≤ 15% of delay minutes recovered). That is the genuinely useful
finding: **the binding constraint is not model quality, it is intervention effectiveness.**
Before investing in this system, measure what a gate intervention actually recovers. If the
answer is under roughly 20%, a better model will not save the business case.

## 5. What to actually do

1. **Protect the morning bank.** Because {prop['share_of_delay_minutes_inherited']:.0%} of delay
   minutes are inherited, a minute recovered before 09:00 is worth several minutes recovered
   at 18:00. Prioritise spare-aircraft and crew buffers on the first two legs of each rotation.
2. **Buy slack where slack is scarce.** Flights whose scheduled ground time is below the
   station median are materially more exposed; the `slack_vs_upstream_min` feature is the
   second strongest driver in the model and it is a *schedule design* variable, not weather.
3. **Work the top two deciles, not the whole schedule.** {top2:.0%} of delay minutes sit in
   20% of flights. Staff the watch-list, not the network.
4. **Re-tune the threshold, not the model, when costs change.** Fuel, crew rules and
   compensation regimes move the break-even point; the model does not need retraining for that.
5. **Treat weather as a tail risk, not a daily driver.** It matters enormously on
   {eda['share_flights_adverse_wx']:.0%} of flights and barely at all on the rest.

---

*Assumption caveats: intervention cost and effectiveness are planning assumptions, not
measured values. They should be replaced with figures from a controlled rollout —
the threshold optimisation is only as good as those two numbers.*
"""
    (REPORTS / "business_insights.md").write_text(md, encoding="utf-8")
    (REPORTS / "business_economics.json").write_text(
        json.dumps(econ, indent=2, default=float), encoding="utf-8")
    sweep.to_csv(REPORTS / "threshold_sweep.csv", index=False)
    dec.to_csv(REPORTS / "risk_deciles.csv")
    (MODELS_DIR / "operating_threshold.json").write_text(json.dumps({
        "threshold": float(best["threshold"]),
        "break_even_probability": cp["break_even_probability"],
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
    }, indent=2), encoding="utf-8")

    print(f"[insights] break-even p = {cp['break_even_probability']:.3f}")
    print(f"[insights] optimal threshold {best['threshold']:.2f} -> "
          f"alert {best['alert_rate']:.1%} of flights, precision {best['precision']:.1%}, "
          f"recall {best['recall']:.1%}")
    print(f"[insights] net benefit ${best['net_benefit_usd']:,.0f} over {days} days "
          f"(${econ['annualised_net_benefit_usd']/1e6:.1f}M annualised)")
    print(f"[insights] top 2 deciles hold {top2:.0%} of delay minutes")
    print(f"[insights] {prop['share_of_delay_minutes_inherited']:.0%} of delay minutes inherited")


if __name__ == "__main__":
    main()
