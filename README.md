# ✈ Flight Delay Prediction — End-to-End Data Science Project

Predicting departure delays across a 35-airport US network: from raw multi-source
data through feature engineering, modelling and explainability, to a served API,
an operations dashboard, and a costed intervention policy.
---

## 1. Overview

| | |
|---|---|
| **Task** | Predict, before departure, whether a flight will push back 15+ minutes late — and by how many minutes |
| **Scale** | 264,870 operated flights · 15 months · 35 airports · 12 carriers · 1,190 routes |
| **Data** | 6 independent sources integrated: flights, hourly weather, airports, aircraft registry, airlines, holiday calendar |
| **Features** | 135 engineered across 6 families → 45 selected by consensus ranking |
| **Models** | 5 classifiers compared + a regression head; LightGBM tuned with Optuna |
| **Classification** | ROC-AUC **0.699** · PR-AUC **0.505** · Brier **0.136** · top-decile lift **2.91×** |
| **Regression** | RMSE **20.28 min** · MAE **12.54 min** · MAPE **67.1%** · R² **0.243** |
| **Delivered as** | FastAPI service + Streamlit dashboard + 19 analytical figures + written reports |

The engineering claim of this project is not the model — it is that **feature
engineering and decision design moved the needle roughly ten times more than model
selection or hyperparameter tuning**, and the numbers below are laid out so you can
check that claim rather than take it on trust.

---

## 2. Problem statement

### The situation, as the data actually describes it

Across the 15 months analysed, **21.4% of departures were 15+ minutes late**. When a
flight is late, the median delay is 35 minutes and the mean is 44. Costed at
industry-standard rates — \$101/min of aircraft direct operating cost plus \$47/min
of passenger time — **each delayed departure destroys roughly \$6,471 of value**.

Airlines can already *react* to delays. What they cannot do is *anticipate* them
precisely enough to act economically. The analysis surfaced four facts that
together define the real problem:

**1. Delay is concentrated, not diffuse.** The top two risk deciles contain **44% of
all delay minutes**. A network-wide response is therefore mostly wasted effort — but
only if you can identify those deciles in advance.

**2. Delay is mostly inherited, not generated.** **32% of all departure-delay minutes
propagate from the aircraft's previous leg** rather than originating at the gate.
Risk compounds through the day: 15.5% on the first leg of a rotation, 21.9% by the
deepest leg; 15.4% at 06:00 rising to **31.0% at 18:00**. A flight is often doomed
hours before anyone looks at its gate.

**3. Weather is a sharp but narrow driver.** Thunderstorms multiply delay risk
**2.27×** (48.3% vs 21.3% baseline), but only **6.2% of flights** meet meaningful
adverse weather, and weather accounts for just **7.3%** of the model's total
explanatory power. Operations teams routinely over-attribute delay to weather.

**4. Blunt policies lose money.** Intervening on every flight costs **−\$25.95M** over
a quarter. A plausible human heuristic ("worse-than-median carrier departing in the
afternoon") costs **−\$1.85M**. Both are worse than doing nothing at all.

### The problem

> Airlines lack a **calibrated, explainable, pre-departure estimate of delay risk for
> each individual flight**, produced early enough and precisely enough to target
> scarce recovery resources — spare aircraft, standby crew, gate and turnaround
> priority — at the small subset of flights where intervention pays for itself,
> rather than spreading them across a schedule where four flights in five will
> depart on time regardless.

### What a solution must satisfy

| Requirement | Why it is non-negotiable |
|---|---|
| **Uses only pre-departure information** | A model that peeks at the actual departure time is worthless in production |
| **Well-calibrated probabilities** | Expected-value decisions need real probabilities, not scores |
| **Ranks reliably** | Operations work a watch-list, so ordering matters more than accuracy |
| **Explains each flight** | A controller will not act on an unexplained number |
| **Carries an explicit cost model** | "Alert at p > 0.5" is an arbitrary choice with a dollar consequence |
| **Evaluated chronologically** | A random split leaks the future into the past and inflates every metric |

---

## 3. Problem breakdown

The headline question decomposes into eight tractable sub-problems.

| # | Sub-problem | Approach | Outcome |
|---|---|---|---|
| 1 | **Data does not exist pre-joined** | Simulate a BTS-calibrated network with real causal structure; integrate 6 sources | 270k flights, 375k weather obs |
| 2 | **Sources disagree on time** | Flights carry local naive timestamps, weather is UTC; align via each airport's zone | 5 zones incl. DST-free Phoenix |
| 3 | **Raw feeds are dirty** | Detect and log duplicates, invalid codes, impossible timestamps, sentinels | 22,297 defects handled |
| 4 | **Raw columns are weak predictors** | Engineer 6 feature families incl. congestion, propagation, rolling history | 135 features |
| 5 | **Leakage is easy and fatal** | Time-windowed aggregates, observability-gated upstream delay, chronological splits | 5 dedicated tests |
| 6 | **Which model?** | Compare 5 classifiers on identical splits; tune with Optuna vs random search | LightGBM shipped |
| 7 | **A probability is not a decision** | Build a cost model; optimise threshold by expected value; test sensitivity | Threshold 0.46, not 0.50 |
| 8 | **Offline features ≠ online features** | Freeze rolling aggregates into a feature store; mirror the transform logic | 1 MB artifact, no skew |

### The two hardest sub-problems

**Leakage (#5).** Three mechanisms guard against it. Rolling aggregates use a
time-based window with `closed="left"`, so a day never contributes to its own
feature. The inbound aircraft's *arrival* delay is used **only if that aircraft
actually landed before our scheduled push-back**; otherwise the model falls back to
the inbound leg's *departure* delay — observable by construction — and flags
`prev_leg_known = 0`. And splits are chronological, never random.

**Train/serve skew (#8).** Rolling features cannot be recomputed for a single future
flight. The feature store freezes the latest state of every aggregate (1,190 routes,
5,848 airport×weekday×hour demand cells, per-airport monthly weather normals) so the
API computes exactly what training computed.

---

## 4. Project architecture and structure

### Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. COLLECT      flights · weather · airports · aircraft ·          │
│                  airlines · holidays                    → data/raw  │
├─────────────────────────────────────────────────────────────────────┤
│  2. CLEAN        dedupe · validate codes · fix timestamps ·         │
│                  LOCAL→UTC alignment · interpolate  → data/interim  │
├─────────────────────────────────────────────────────────────────────┤
│  3. INTEGRATE    6 tables → 1 flight-level frame (76 cols)          │
├─────────────────────────────────────────────────────────────────────┤
│  4. EDA          10 figures + written summary            → reports/ │
├─────────────────────────────────────────────────────────────────────┤
│  5. FEATURES     time · weather · airport · flight ·                │
│                  geospatial · historical → 135     → data/processed │
├─────────────────────────────────────────────────────────────────────┤
│  6. SELECT       MI + LightGBM gain + RFE consensus → 45 features   │
├─────────────────────────────────────────────────────────────────────┤
│  7. TRAIN        LogReg · RandomForest · XGBoost ·                  │
│                  LightGBM · CatBoost                     → models/  │
├─────────────────────────────────────────────────────────────────────┤
│  8. TUNE         Optuna TPE vs RandomizedSearchCV ·                 │
│                  calibration probe   → models/final_model.joblib    │
├─────────────────────────────────────────────────────────────────────┤
│  9. STORE        freeze rolling aggregates    → feature_store.json  │
├─────────────────────────────────────────────────────────────────────┤
│ 10. EXPLAIN      SHAP global · dependence · per-flight narratives   │
├─────────────────────────────────────────────────────────────────────┤
│ 11. REGRESS      delay-minutes head → RMSE / MAE / MAPE             │
├─────────────────────────────────────────────────────────────────────┤
│ 12. INSIGHTS     cost model · threshold optimisation · sensitivity  │
└───────────────┬─────────────────────────────┬───────────────────────┘
                ▼                             ▼
        ┌───────────────┐            ┌──────────────────┐
        │  FastAPI      │            │  Streamlit       │
        │  :8000        │            │  :8501           │
        │  /predict     │            │  7 views, live   │
        │  /delay-risk  │            │  scoring + SHAP  │
        └───────────────┘            └──────────────────┘
```

### Repository structure

```
flight-delay-prediction/
│
├── config/config.yaml              every tunable parameter; no magic numbers in code
│
├── src/
│   ├── config.py                   config loader + path constants
│   ├── timezones.py                cross-platform IANA tz handling (Windows-safe)
│   │
│   ├── data/
│   │   ├── reference.py            35 real airports, 12 carriers, 14 aircraft types
│   │   ├── generate_synthetic.py   causal network simulator + defect injection
│   │   ├── clean.py                validation, tz alignment, quality report
│   │   └── ingest.py               6-source integration
│   │
│   ├── features/
│   │   ├── build_features.py       6 feature families, leakage-controlled
│   │   └── selection.py            MI + gain + RFE consensus
│   │
│   ├── analysis/
│   │   ├── style.py                shared visual language
│   │   └── eda.py                  10 figures + written summary
│   │
│   ├── models/
│   │   ├── dataset.py              chronological splits, dtype handling
│   │   ├── train.py                5-model comparison (resumable)
│   │   ├── tune.py                 Optuna + calibration probe + final bundle
│   │   ├── train_regression.py     delay-minutes head (RMSE/MAE/MAPE)
│   │   ├── explain.py              SHAP + human-readable narration
│   │   ├── feature_store.py        offline→online aggregate freeze
│   │   └── featurize_online.py     serving-time twin of build_features
│   │
│   └── insights/business.py        cost model, threshold optimisation, sensitivity
│
├── api/main.py                     FastAPI: /health /model-info /predict /delay-risk
├── dashboard/app.py                Streamlit: 7 views
├── tests/test_pipeline.py          24 tests, incl. 5 leakage guards
├── run_pipeline.py                 orchestrator with preflight checks
│
├── reports/                        figures/ + metrics + written analyses
├── models/                         trained artifacts + feature store
└── data/                           raw / interim / processed (regenerated, not shipped)
```

---

## 5. How the project works

### Stage 1–2 · Collection and cleaning

Six independent sources are produced, then integrated. Flight records are
**simulated**, because no public dataset ships flights, weather, fleet and holidays
pre-joined — but the simulation encodes real causal structure, not noise: aircraft
fly **multi-leg rotations** so delay propagates; airport congestion is **endogenous**
(scheduled demand vs. declared capacity); weather severity derives from raw
observations the model never sees. It is calibrated to published BTS figures: 21.4%
late, 1.3% cancelled, 35-minute median delay.

Realistic defects are injected on purpose, and every one is caught and logged to
`reports/data_quality_report.md`:

| Defect | Caught |
|---|---:|
| Duplicate flight IDs | 1,077 |
| Invalid airport codes (`ZZZ`, `N/A`, `---`) | 403 |
| Arrival timestamp before departure | 195 |
| Weather sentinel values (`-999`, `99999`) | 8,988 |
| Missing weather hours → bounded interpolation | 8,529 |
| Duplicate weather observations | 749 |
| Sentinel distances / absurd delays | 751 |
| Missing tail numbers (flagged, not dropped) | 1,610 |

**Time-zone alignment is the subtle one.** Flight times arrive as local naive
strings; weather is UTC. Joining them naively misaligns most of the network
*silently*. A regression test pins 08:00 Chicago to 14:00 UTC in January.

### Stage 5 · Feature engineering — where the project is won

| Family | n | Representative features |
|---|---:|---|
| **Time** | 30 | cyclical hour/day encodings, holiday proximity, bank flags, red-eye, season |
| **Weather** | 31 | intensity bands, IFR/LIFR flags, gust factor, dewpoint spread, temperature anomaly, composite severity |
| **Airport** | 15 | demand ÷ declared capacity, 3-hour bank pressure, carrier dominance, runways per departure |
| **Flight** | 19 | **schedule padding**, turnaround buffer, rotation leg depth, airframe age, implied speed |
| **Geospatial** | 20 | great-circle distance, bearing, region pair, hub-to-hub, elevation gain |
| **Historical** | 20 | rolling carrier 7d / route 30d / station 7d rates, **upstream propagation**, ground-time slack |

### Stage 6 · Selection

Consensus of mutual information, LightGBM gain and RFE, after correlation pruning at
|r| > 0.95 — fitted on the **training window only**, since selecting on the full
dataset leaks the test window into the feature set itself.

**45 features survive**, and the mix vindicates the effort: the top three are
`upstream_delay_min`, `slack_vs_upstream_min` and `aircraft_cum_delay_today`.
**No raw column reaches the top five.**

### Stage 9–10 · Explainability and decision

SHAP produces global attribution, dependence plots and per-flight narratives in
operational language, e.g.:

> Predicted departure-delay risk 96%. Driven up by: inbound aircraft late (65 min);
> weather severity across the route (4.2 index); ground-time slack (−50 min).

The threshold is then set by **expected value**, not convention — see §8.

---

## 6. How to run

### Requirements

Python 3.10+. On **Windows**, `tzdata` is mandatory (Windows ships no IANA timezone
database); it is already listed in `requirements.txt`.

```bash
git clone <repo> && cd flight-delay-prediction
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

### Build the data

The dataset is **not shipped** (~65 MB) but is fully reproducible from a fixed seed —
regenerating gives byte-identical data, so the shipped models stay valid.

```bash
python run_pipeline.py --from collect --to features   # ~5 min
python run_pipeline.py --only eda                     # optional figures
```

### Or run everything

```bash
python run_pipeline.py                            # all 12 stages, ~20 min
python run_pipeline.py --list                     # show stages
python run_pipeline.py --from tune --to insights   # resume a range
```

`run_pipeline.py` preflights the timezone database and each stage's inputs, and
names the exact command to run if something is missing.

### Serve

```bash
uvicorn api.main:app --port 8000       # http://localhost:8000/docs
streamlit run dashboard/app.py         # http://localhost:8501
pytest tests/ -v                       # 24 tests
```

The API needs **no dataset** — model and feature store are in the repository. The two
services are independent; run either or both, in separate terminals.

### Example request

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "carrier_code":"MQ","origin":"ORD","destination":"LGA",
  "scheduled_departure_local":"2024-04-12T18:20:00",
  "prev_leg_arr_delay_min":65,"scheduled_ground_time_min":45,
  "origin_weather":{"precip_mm":9.0,"wind_gust_kt":41,"visibility_km":2.0,
                    "cloud_ceiling_ft":600,"condition":"Thunderstorm"}}'
```
```json
{ "delay_probability": 0.961, "risk_band": "high",
  "recommend_intervention": true, "expected_delay_minutes": 42.3 }
```

The same flight in clear weather with an on-time inbound scores **0.095**.

---

## 7. Training details and evaluation metrics

### Setup

| | |
|---|---|
| **Split** | Chronological. Train ≤ 2023-10-31 (177,298) · Valid → 2023-12-31 (35,069) · Test = Q1 2024 (52,591) |
| **Why not random?** | A random split lets future days inform past predictions through the rolling history features; it would inflate every number here by an estimated 2–4 points |
| **Class balance** | 20.9% positive on test — imbalanced, so PR-AUC leads over accuracy |
| **Early stopping** | On the validation window; the test window is touched exactly once |

### Classification — model comparison

Identical splits, identical features, selected on validation PR-AUC.

| Model | Valid ROC-AUC | Valid PR-AUC | Brier | Test ROC-AUC | Test PR-AUC | Fit |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | **0.7128** | **0.5465** | 0.1428 | 0.7007 | 0.5071 | 113 s |
| **LightGBM** *(shipped)* | 0.7100 | 0.5424 | 0.1430 | 0.6976 | 0.5039 | **11 s** |
| XGBoost | 0.7097 | 0.5420 | 0.1432 | 0.6978 | 0.5031 | 19 s |
| Random Forest | 0.7051 | 0.5298 | 0.1463 | 0.6944 | 0.4938 | 77 s |
| Logistic Regression | 0.6997 | 0.5177 | 0.1487 | 0.6839 | 0.4732 | 5 s |

**The spread from a logistic baseline to the best model is 1.3 ROC-AUC points.** That
is the most informative number in the table: model choice mattered far less than
feature construction. CatBoost edges LightGBM by 0.3pp at ten times the training
cost; LightGBM ships.

### Hyperparameter optimisation

| Strategy | Budget | Result |
|---|---|---|
| RandomizedSearchCV | 8 candidates × 2 folds, 94 s | CV PR-AUC 0.4935 |
| **Optuna TPE** | 10 trials, 110 s | **Valid PR-AUC 0.5438** |

Tuning bought ≈0.15 pp over default LightGBM — reported plainly because it is true.

### Calibration was tested, then rejected

Isotonic regression was fitted on half the validation window and judged on the other
half:

| | ECE | Brier | PR-AUC |
|---|---:|---:|---:|
| **Raw LightGBM** | **0.0111** | 0.1479 | **0.5595** |
| + isotonic | 0.0138 | 0.1481 | 0.5455 |

The raw model was *already better calibrated* than the correction, and isotonic's
step function cost 1.4 pp of ranking resolution. The pipeline detects this
automatically and ships uncalibrated. Applying calibration reflexively — the default
habit — would have made the model worse.

### Final classification metrics (test window, 52,591 flights)

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6989 |
| PR-AUC | 0.5047 |
| Brier score | 0.1363 |
| Expected calibration error | 0.0111 |
| Log loss | 0.4379 |
| Accuracy @ 0.46 | 82.9% |
| Precision @ 0.46 | 89.9% |
| Recall @ 0.46 | 20.8% |
| **Top-decile lift** | **2.91×** |

### Regression head — RMSE, MAE, MAPE_pct

The classifier answers *"will it be late?"*. A second LightGBM head answers *"by how
many minutes?"* — and this is what RMSE / MAE / MAPE describe.

**Objective choice matters, so both were fitted:**

| Objective | RMSE (min) | MAE (min) | MAPE_pct | R² |
|---|---:|---:|---:|---:|
| L1 (MAE-optimal) | 21.56 | **10.45** | 84.4 | 0.145 |
| **L2 (RMSE-optimal)** *(shipped)* | **20.28** | 12.54 | **67.1** | **0.243** |

L2 wins on three of four metrics, so it ships — but if MAE is your operating metric,
the L1 model is in the same module and is 16.7% better on it.

**Shipped regression model, test window:**

| Metric | Value | Naive median baseline | Δ |
|---|---:|---:|---:|
| **RMSE** | **20.28 min** | 24.70 min | **−17.9%** ✅ |
| **MAE** | **12.54 min** | 12.25 min | +2.3% ⚠️ |
| **MAPE_pct** | **67.1%** | 100.0% | **−32.9 pp** ✅ |
| WAPE_pct | 102.3% | 100.0% | +2.3% |
| R² | 0.243 | −0.121 | ✅ |
| Median absolute error | 8.54 min | 4.00 min | |

**Two caveats stated openly, because both look bad if discovered rather than
disclosed:**

**MAPE needs a restricted denominator.** Departure delay is signed and often near
zero — flights push back early. MAPE divides by the actual value, so on the full
population it is undefined for on-time flights and explodes for near-zero ones:
computed naively it reads **227%**, which is noise, not signal. The 67.1% above is
computed over **delayed flights only (actual ≥ 15 min, n = 11,015)**, where the
denominator is well-posed. **WAPE** is reported alongside because it is defined
everywhere. Any single unqualified "MAPE" figure on this target should be distrusted.

**MAE is marginally worse than a constant.** The naive baseline predicts the training
median for every flight, and the median is *MAE-optimal by construction* on a
right-skewed target. Beating it on MAE requires the L1 model (10.45 min, −14.7%). The
L2 model wins decisively on RMSE, MAPE and R² instead. This is a metric-selection
artefact rather than a broken model — but quoting MAE without its baseline would
mislead in either direction.

**The honest summary:** delay *magnitude* is much harder to predict than delay
*occurrence* (R² 0.243), because the tail is driven by cascading disruptions
unknowable hours ahead. Classification is the more useful framing, which is why it is
the primary model.

---

## 8. Business insights

### Ranking beats classifying

| Risk decile | Flights | Delay rate | Lift | Share of delay minutes | Cumulative |
|---|---:|---:|---:|---:|---:|
| **D10** | 5,259 | **60.9%** | **2.91×** | **31.7%** | 31.7% |
| D9 | 5,259 | 25.4% | 1.21× | 12.6% | **44.3%** |
| D8 | 5,259 | 22.3% | 1.06× | 10.6% | 54.9% |
| D7 | 5,259 | 20.1% | 0.96× | 9.1% | 64.0% |
| D5 | 5,259 | 16.5% | 0.79× | 7.5% | 79.5% |
| D1 | 5,260 | 8.0% | 0.38× | 3.6% | 100% |

**Work the top two deciles, not the network: 20% of flights hold 44% of delay minutes.**

### Where delay comes from

- **32% of all delay minutes are inherited** from the previous leg, not generated at
  the gate. With no upstream delay the rate is 17.3%; with 60–120 minutes inherited
  it is **86.5%**.
- **The day compounds:** 15.5% on the first leg → 21.9% by the deepest leg; 15.4% at
  06:00 → **31.0% at 18:00**. Worst weekly slot: **Friday 18:00 at 37.2%**.
- **Weather is narrow:** a 2.27× multiplier, but only 6.2% of flights affected and
  only **7.3%** of model explanatory power.
- **Concentration:** ORD alone generates **9.8%** of network delay minutes; the top
  three stations **22.0%**.
- **Carrier spread is 15.6 pp:** Delta 14.4% vs Envoy 29.9%. Regionals are
  systematically worse — short turnarounds, deeper rotations.
- **Holiday windows** (±3 days of a major travel holiday) run **+39.9%** above normal.

### The threshold is an economics question, not an ML question

| Parameter | Value |
|---|---:|
| Aircraft direct operating cost | \$101/min |
| Passenger time cost | \$47/min |
| Mean delay when late | 43.7 min |
| **Cost of one delayed departure** | **\$6,471** |
| Cost of one intervention | \$900 |
| Delay minutes recovered by intervening | 30% |
| Benefit of one correct intervention | \$1,941 |

The cost asymmetry is 2.2:1, so the **break-even probability is 0.46, not 0.50**.

| Policy | Net benefit (92-day test window) |
|---|---:|
| Intervene on every flight | **−\$25.95M** |
| Heuristic: worse-than-median carrier, afternoon departure | **−\$1.85M** |
| **Model at p ≥ 0.46** | **+\$2.15M** |

At the optimum: alert on **5.1%** of flights, precision **87.6%**, recall **21.4%** —
**\$8.5M annualised** at this traffic level.

### The assumptions are stress-tested

A 15-cell sensitivity grid over intervention cost and effectiveness shows the optimal
alert rate swinging from **2% to 91%** of flights, and net benefit turning **negative
in 3 of 15 scenarios** — when interventions are both expensive (≥\$1,500) and weak
(≤15% recovery).

> **The binding constraint is intervention effectiveness, not model quality.** Before
> investing in this system, measure what a gate intervention actually recovers. Below
> roughly 20%, a better model will not save the business case.

### Recommended actions

1. **Protect the morning bank** — a minute recovered before 09:00 is worth several recovered at 18:00.
2. **Buy slack where slack is scarce** — `slack_vs_upstream_min` is the #2 driver and is a *schedule design* variable, not weather.
3. **Staff the watch-list, not the network** — 20% of flights carry 44% of the pain.
4. **Re-tune the threshold, not the model, when costs change** — fuel, crew rules and compensation regimes move break-even; no retraining needed.
5. **Treat weather as tail risk** — decisive on 6% of flights, near-irrelevant on the rest.


## 9. Future improvements

### Correctness first — do these before anything else

1. **Replace observed weather with forecasts.** Training attaches the observation
   valid at the scheduled hour; production must attach the **TAF forecast available
   at decision time**. This will *lower* measured performance, and that lower number
   is the honest one. Join key and column names are identical — one line in
   `attach_weather`.
2. **Swap in real data.** Point the loader at BTS On-Time Performance + NOAA ISD.
   Every downstream stage reads the same six filenames and schemas.
3. **Measure intervention effectiveness.** The single highest-value experiment here:
   a controlled rollout measuring what a gate intervention actually recovers. The
   sensitivity grid shows the entire business case hinges on it.

### Modelling

4. **Model the cascade explicitly.** Delay propagates along aircraft rotations — a
   graph structure the current model sees only through summary features. A temporal
   graph network over the rotation DAG should capture second-order propagation the
   tabular model misses.
5. **Quantile regression for the tail.** R² 0.243 reflects genuine difficulty, but
   quantile objectives (P50/P90) would give operations a *range* rather than a point
   estimate — far more actionable for spare-aircraft decisions.
6. **Survival framing.** Model *time-to-departure* rather than a binary at a fixed
   threshold; this handles "how late" and "will it be late" in one object and
   supports dynamic updates as the departure hour approaches.
7. **Multi-horizon predictions.** Re-score at T−24h, T−4h, T−1h. Accuracy rises as
   the inbound aircraft's status resolves, and the horizon curve tells operations
   *when* a decision becomes reliable.
8. **Predict cancellations jointly.** Currently excluded (a cancelled flight has no
   departure delay). A multi-task head sharing the encoder would be more useful and
   more statistically efficient.

### Production engineering

9. **Drift monitoring.** Feature-distribution and calibration monitoring with
   alerting. The feature store already carries an `as_of` date as the hook.
10. **Automated retraining** on a rolling window, with champion/challenger evaluation
    gated on chronological backtests.
11. **Have the dashboard call the API.** It currently scores in-process — fine for a
    single-machine demo, but production should have exactly one scoring path.
12. **Containerise** (`Dockerfile` + `docker-compose`) and add CI running the test
    suite plus a leakage regression check on every commit.
13. **Feature store proper.** The frozen JSON works at this scale; a real deployment
    wants Feast or equivalent with point-in-time correctness guarantees.

### Analytical extensions

14. **Network effects.** Model delay at the *station* level — when a hub degrades,
    every downstream flight is affected simultaneously, and per-flight independence
    understates correlated risk.
15. **Counterfactual schedule design.** The model can score hypothetical schedules,
    turning it from a prediction tool into a **planning** tool: what does adding 10
    minutes of ground time at ORD do to network-wide delay minutes?
16. **Crew and passenger connection modelling.** Misconnects are where delay converts
    into real customer cost; the current cost model approximates this with a flat
    per-minute rate.

---

## Known limitations

1. **Flight data is simulated** — structurally faithful and BTS-calibrated, but the
   absolute numbers describe a synthetic network. Swap in real data before quoting
   any figure externally.
2. **Weather uses observations, not forecasts** — see improvement #1.
3. **Intervention cost and effectiveness are assumptions**, not measurements; the
   sensitivity grid quantifies exactly how much that matters.
4. **Cancellations are excluded** from the modelling frame.
5. **No drift monitoring** in the current deployment.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: features.parquet` | dataset not built (it is not shipped) | `python run_pipeline.py --from collect --to features` |
| `ZoneInfoNotFoundError` | Windows has no system IANA database | `pip install tzdata` |
| Dashboard: "Dataset not built" | as row 1 | as above; Risk predictor and Model views work meanwhile |

## Testing

```
24 passed in 6.6s
```

Geometry, weather-severity monotonicity, all cleaning defect classes, time-zone
alignment, cyclical encodings, API contracts — and **five leakage guards**:
unobservable inbound arrivals, rotation-chain breaks, chronological split
disjointness, and target-derived columns in the feature set.
