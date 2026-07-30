# Business Insights

*Generated from the test window (2024-01-01 → 2024-04-01,
52,591 flights) using the calibrated production model.*

---

## 1. The headline is not the accuracy

The model scores 0.699 ROC-AUC. On its own that number
is operationally meaningless. What matters is that **the top two risk deciles contain
44% of all delay minutes** in the network. Ranking flights by predicted risk and
working the top 20% reaches most of the pain with a fifth of the effort.

| Risk decile | Flights | Delay rate | Lift | Share of delay minutes |
|---|---:|---:|---:|---:|
| D10 | 5,259 | 60.8% | 2.90x | 31.7% |
| D9 | 5,259 | 25.7% | 1.23x | 12.5% |
| D8 | 5,259 | 22.5% | 1.08x | 10.7% |
| D7 | 5,259 | 19.8% | 0.94x | 9.2% |
| D6 | 5,259 | 17.6% | 0.84x | 7.8% |
| D5 | 5,259 | 15.7% | 0.75x | 7.2% |
| D4 | 5,259 | 14.8% | 0.71x | 6.8% |
| D3 | 5,259 | 12.7% | 0.61x | 5.6% |
| D2 | 5,259 | 11.7% | 0.56x | 5.0% |
| D1 | 5,260 | 8.1% | 0.38x | 3.6% |

## 2. Where delay actually comes from

- **Propagation is the dominant mechanism.** 32% of all
  departure-delay minutes are *inherited* from the aircraft's previous leg rather than
  generated at the gate. 33% of delayed flights had an
  inbound aircraft that was already more than 15 minutes late.
- **The day compounds.** First flight of the day: 15.5% delay rate;
  by the deepest leg of the rotation it is 21.9%. Recovery has to happen
  early or not at all.
- **Weather is sharp but narrow.** Thunderstorms multiply delay risk by
  2.3x, but only 6% of flights face
  meaningful adverse weather. Weather explains
  8% of the model's total explanatory power —
  real, but far less than operations staff usually assume.
- **Congestion is structural.** Delay risk rises monotonically with the departure-bank
  demand/capacity ratio at the origin.

## 3. Concentration: a few places and times carry the network

- ORD alone generates 9.8%
  of all network delay minutes; the top three stations account for 22.0%.
- The worst weekly slot is **Fri 18:00 (37.2%)**, against a system average of
  21.4%.
- Holiday windows (±3 days of a major travel holiday) run **+40%**
  above normal.
- Carrier spread is 16 percentage points between best
  (Delta Air Lines (14.4%)) and worst (Envoy Air (29.9%)). At peak hours the best performer is
  **Delta Air Lines** (14.3%) and the worst is
  **SkyWest Airlines** (29.6%).

## 4. The alerting threshold is an economics question, not an ML question

Cost assumptions (editable in `config/config.yaml`):

| Parameter | Value |
|---|---:|
| Aircraft direct operating cost | $101/min |
| Passenger time cost | $47/min |
| Average delay when a flight is late | 44 min |
| Cost of one delayed departure | **$6,471** |
| Cost of one intervention | $900 |
| Delay minutes recovered by intervening | 30% |
| Benefit of one correct intervention | $1,941 |

A false positive costs $900. A missed delay costs
$1,941 in forgone recovery. The asymmetry is
2.2:1, so the
break-even probability is **0.46**, not 0.50.

Sweeping the threshold over the test window gives an optimum at
**p ≥ 0.53**:

- alert on **4.7%** of flights (2,451 of 52,591)
- precision **91.6%**, recall **20.4%**
- net benefit **$2,154,329** over 92 days
  ($40.96 per flight flown)
- annualised at this traffic level: **$8.5M**

### Versus the alternatives

| Policy | Net benefit (test window) |
|---|---:|
| Intervene on every flight | $-25,953,964 |
| Heuristic (worse-than-median carrier, afternoon departure) | $-1,854,064 |
| **Model at optimal threshold** | **$2,154,329** |

The model beats the plausible human heuristic by
**$4,008,393** over three months.

### How much does this depend on my assumptions?

Intervention cost and effectiveness are *assumptions*, not measurements, and the
threshold moves with them. The honest version of the answer is the grid:

| Intervention cost | Effectiveness | Break-even p | Optimal threshold | Alerts | Precision | Recall | Net benefit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| $300 | 15% | 0.31 | 0.31 | 8.6% | 66% | 27% | $1,540,254 |
| $300 | 30% | 0.15 | 0.16 | 55.1% | 28% | 74% | $7,087,203 |
| $300 | 50% | 0.09 | 0.09 | 95.0% | 22% | 98% | $20,058,895 |
| $600 | 15% | 0.62 | 0.59 | 4.4% | 94% | 20% | $724,173 |
| $600 | 30% | 0.31 | 0.31 | 8.6% | 66% | 27% | $3,080,508 |
| $600 | 50% | 0.19 | 0.18 | 42.8% | 31% | 64% | $9,285,332 |
| $900 | 15% | 0.93 | 0.88 | 3.2% | 99% | 15% | $103,701 |
| $900 | 30% | 0.46 | 0.53 | 4.7% | 92% | 20% | $2,154,329 |
| $900 | 50% | 0.28 | 0.28 | 11.3% | 57% | 31% | $5,670,860 |
| $1,500 | 15% | 1.55 | 0.95 | 2.5% | 100% | 12% | $-696,252 |
| $1,500 | 30% | 0.77 | 0.70 | 3.9% | 97% | 18% | $793,810 |
| $1,500 | 50% | 0.46 | 0.53 | 4.7% | 92% | 20% | $3,590,549 |
| $2,500 | 15% | 2.58 | 0.95 | 2.5% | 100% | 12% | $-2,004,252 |
| $2,500 | 30% | 1.29 | 0.95 | 2.5% | 100% | 12% | $-738,504 |
| $2,500 | 50% | 0.77 | 0.70 | 3.9% | 97% | 18% | $1,323,016 |

Two things stand out, and the second one matters more than the first.

**The policy is very sensitive.** The optimal alert rate ranges from
2% to 95% of flights across this grid.
Anyone quoting a single headline threshold without stating the cost assumptions behind it
is quoting noise.

**The programme is not universally worth running.** Net benefit is positive in
12 of 15 scenarios but turns *negative* in
3 — specifically when interventions are both expensive
(≥ $1,500) and weak (≤ 15% of delay minutes recovered). That is the genuinely useful
finding: **the binding constraint is not model quality, it is intervention effectiveness.**
Before investing in this system, measure what a gate intervention actually recovers. If the
answer is under roughly 20%, a better model will not save the business case.

## 5. What to actually do

1. **Protect the morning bank.** Because 32% of delay
   minutes are inherited, a minute recovered before 09:00 is worth several minutes recovered
   at 18:00. Prioritise spare-aircraft and crew buffers on the first two legs of each rotation.
2. **Buy slack where slack is scarce.** Flights whose scheduled ground time is below the
   station median are materially more exposed; the `slack_vs_upstream_min` feature is the
   second strongest driver in the model and it is a *schedule design* variable, not weather.
3. **Work the top two deciles, not the whole schedule.** 44% of delay minutes sit in
   20% of flights. Staff the watch-list, not the network.
4. **Re-tune the threshold, not the model, when costs change.** Fuel, crew rules and
   compensation regimes move the break-even point; the model does not need retraining for that.
5. **Treat weather as a tail risk, not a daily driver.** It matters enormously on
   6% of flights and barely at all on the rest.

---

*Assumption caveats: intervention cost and effectiveness are planning assumptions, not
measured values. They should be replaced with figures from a controlled rollout —
the threshold optimisation is only as good as those two numbers.*
