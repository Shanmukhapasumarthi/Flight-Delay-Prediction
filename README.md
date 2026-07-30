# Flight Delay Prediction — End-to-End

Predicting whether a flight will depart 15+ minutes late, from data collection
through to a deployed API, a dashboard, and a costed operational decision.

The point of this project is not the model. The model is 30 lines. The point is
everything around it: integrating six data sources, engineering features that
respect what is knowable at prediction time, and converting a probability into a
decision that has a dollar value attached.

```bash
pip install -r requirements.txt
python run_pipeline.py                 # ~20 min end to end
uvicorn api.main:app --port 8000       # http://localhost:8000/docs
streamlit run dashboard/app.py         # http://localhost:8501
pytest tests/ -v                       # 24 tests
```

---

## Results at a glance

| | |
|---|---|
| Flights modelled | 264,870 (15 months, 35 airports, 12 carriers, 1,190 routes) |
| Features engineered → selected | 135 → 45 |
| Best model | LightGBM, Optuna-tuned |
| Test ROC-AUC / PR-AUC | **0.699 / 0.505** (base rate 20.9%) |
| Brier / ECE | 0.136 / 0.011 |
| Top-decile lift | **2.91×** — captures 31.7% of all delay minutes |
| Optimal alerting threshold | **0.46**, not 0.50 (set by cost asymmetry) |
| Net benefit at that threshold | $2.15M over 92 days; **$8.5M annualised** |

The honest read on 0.699 ROC-AUC: departure delay has a large irreducible random
component, and this is a **chronological** split, which is 2–4 points harsher than
the random splits that produce the 0.90+ numbers commonly reported on this problem.
A random split here would leak the future into the past through the rolling
history features and inflate every number below.

---

## 1. Business framing

Four questions, answered with numbers in `reports/business_insights.md`:

| Question | Answer from this data |
|---|---|
| Which airlines are worst? | 14.7pp spread — Delta 14.0% vs Envoy 28.8%. Regionals are systematically worse. |
| Which airports drive delay? | ORD alone contributes 9.4% of network delay minutes; top 3 = 21.8%. |
| Which weather matters? | Thunderstorms are a 2.5× multiplier, but hit only ~6% of flights. |
| How do predictions cut cost? | Ranking flights by risk puts 44% of delay minutes in the top 2 deciles. |

## 2. Data collection — six sources, deliberately separate

| Source | Rows | Contents |
|---|---:|---|
| `flights.csv` | 270,323 | operational records, **local naive** timestamps |
| `weather_hourly.csv` | 375,260 | hourly METAR-style obs, **UTC** |
| `airports.csv` | 35 | real coordinates, runways, elevation, tz, declared capacity |
| `aircraft.csv` | 168 | tail-level registry: model, seats, age |
| `airlines.csv` | 12 | carrier reference |
| `holiday_calendar.csv` | 19 | US federal + travel-heavy dates |

Flight records are **simulated** (`src/data/generate_synthetic.py`), because no
public dataset ships flights, weather, fleet and holidays pre-joined. The
simulation encodes real causal structure rather than random noise:

- aircraft fly **multi-leg rotations**, so delay propagates down the day
- airport congestion is **endogenous** — scheduled demand against declared capacity
- weather severity is derived from raw observations **the model never sees**
- carriers differ in operational quality; holidays and evening banks matter

Calibrated against published BTS figures: 21.4% of departures late ≥15 min, 1.3%
cancelled, 35-minute median delay when late.

> **If you have real data:** replace `generate_synthetic.py` with a loader for BTS
> On-Time Performance + NOAA ISD. Every downstream stage reads the same six
> filenames and schemas, so nothing else changes.

## 3. Cleaning — defects are injected on purpose, then caught

Every defect below is deliberately written into the raw files and detected by
`src/data/clean.py`, which logs every action to `reports/data_quality_report.md`.

| Defect | Caught |
|---|---:|
| Duplicate flight IDs | 1,077 |
| Invalid airport codes (`ZZZ`, `N/A`, `---`) | 403 |
| Arrival timestamp before departure | 195 |
| Sentinel values (`-999`, `99999`) in weather | 8,988 |
| Missing weather hours → bounded interpolation | 8,529 |
| Duplicate weather observations | 749 |
| Sentinel distances / absurd delays | 751 |
| Missing tail numbers (flagged, not dropped) | 1,610 |

**Time-zone alignment is the subtle one.** Flight times arrive as local naive
strings; weather is UTC. Joining them naively silently misaligns most of the
network. Cleaning localises each timestamp using its *airport's* zone across 5
distinct zones — including Phoenix, which does not observe DST — and there is a
regression test pinning 08:00 Chicago to 14:00 UTC in January.

## 4. EDA

Ten figures in `reports/figures/`, summarised in `reports/eda_summary.md`.

- Delay risk **compounds through the day**: 14.8% at 06:00 → 30.4% at 18:00
- Worst weekly slot: **Friday 18:00, 35.8%**
- Holiday windows (±3 days) run **+41%**
- Rate and impact rank differently — small stations can be chronically late and
  irrelevant to the network

## 5. Feature engineering — six families, 135 features

| Family | n | Examples |
|---|---:|---|
| Time | 30 | cyclical hour/day encodings, holiday proximity, bank flags, season |
| Weather | 31 | intensity bands, IFR/LIFR flags, gust factor, temperature anomaly, composite severity |
| Airport | 15 | demand ÷ declared capacity, 3-hour bank pressure, carrier dominance |
| Flight | 19 | **schedule padding**, turnaround buffer, leg depth, airframe age |
| Geospatial | 20 | great-circle distance, bearing, region pair, hub-to-hub, elevation gain |
| Historical | 20 | rolling carrier 7d / route 30d / station 7d rates, **upstream propagation** |

### Leakage control — the part that decides whether any of this is real

1. **Rolling aggregates use `closed="left"`** on a time-based window, so a day
   never contributes to its own feature.
2. **Upstream delay is gated on observability.** The inbound leg's *arrival*
   delay is used only if that aircraft actually landed before our scheduled
   push-back; otherwise the model falls back to the inbound leg's *departure*
   delay — which was observable — and sets `prev_leg_known = 0`. Three tests in
   `tests/test_pipeline.py` enforce this, including one that breaks the rotation
   chain when the previous leg landed at a different station.
3. **Chronological splits** — train ≤ 2023-10-31, validation → 2023-12-31,
   test = Q1 2024, never touched until the final evaluation.

## 6. Feature selection

Consensus ranking across mutual information, LightGBM gain, and RFE, after
correlation pruning at |r| > 0.95 (11 redundant features removed, e.g.
`distance_km` ↔ `scheduled_block_min` at r = 0.999). Fitted on the training
window only — selecting on the full data set leaks the test window into the
feature set itself.

**45 features survive.** The mix vindicates the engineering effort: 12 historical,
9 time, 9 weather, 8 flight, 5 geospatial, 2 airport. The top three are
`upstream_delay_min`, `slack_vs_upstream_min` and `aircraft_cum_delay_today` —
**not one raw column makes the top five.**

## 7. Model comparison

Identical splits, identical features. Selection metric is PR-AUC, because the
target is imbalanced and the operational question is a precision/recall
trade-off, not an accuracy question.

| Model | Valid ROC-AUC | Valid PR-AUC | Brier | Test ROC-AUC | Test PR-AUC | Fit |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | **0.7128** | **0.5465** | 0.1428 | 0.7007 | 0.5071 | 113s |
| LightGBM | 0.7100 | 0.5424 | 0.1430 | 0.6976 | 0.5039 | **11s** |
| XGBoost | 0.7097 | 0.5420 | 0.1432 | 0.6978 | 0.5031 | 19s |
| Random Forest | 0.7051 | 0.5298 | 0.1463 | 0.6944 | 0.4938 | 77s |
| Logistic Regression | 0.6997 | 0.5177 | 0.1487 | 0.6839 | 0.4732 | 5s |

The spread between the best model and a logistic baseline is **1.3 ROC-AUC
points**. The spread between raw columns and engineered features is far larger.
That ordering is the main empirical lesson of the project.

## 8. Hyperparameter optimisation

Two strategies on the same estimator, so the comparison is about the search:

| Strategy | Budget | Result |
|---|---|---|
| RandomizedSearchCV | 8 candidates × 2 folds, 94s | CV PR-AUC 0.4935 |
| Optuna TPE | 10 trials, 110s | **Validation PR-AUC 0.5438** |

Tuning bought ~0.15pp over default LightGBM. Reported plainly because it is the
truth: on this problem, feature engineering was worth roughly ten times more than
hyperparameter search.

### Calibration was tested, then rejected

Isotonic calibration was fitted on half the validation window and judged on the
other half:

| | ECE | Brier | PR-AUC |
|---|---:|---:|---:|
| Raw LightGBM | **0.0111** | 0.1479 | **0.5595** |
| + isotonic | 0.0138 | 0.1481 | 0.5455 |

The raw model was *already better calibrated* than the correction, and isotonic's
step function destroyed ranking resolution. The pipeline detects this
automatically and ships the uncalibrated model. Applying calibration reflexively
would have cost 1.4pp of PR-AUC.

## 9. Explainability

`src/models/explain.py` produces global attribution, dependence plots, and
per-flight explanations translated into operational language.

**Share of total explanatory power by family:**

| Historical | Time | Flight | Weather | Geospatial | Airport |
|---:|---:|---:|---:|---:|---:|
| 36.2% | 26.4% | 24.5% | 7.3% | 5.2% | 0.4% |

Weather at 7.3% is the counter-intuitive result and it is worth stating carefully:
weather is a *sharp but narrow* driver — a 2.5× multiplier on the ~6% of flights
that meet it. (The airport family reads low because only two airport features
survived selection; station identity is carried by the `origin`/`destination`
categoricals inside the geospatial family.)

A real explanation from the API:

> Predicted departure-delay risk 96%. Driven up by: inbound aircraft late (65
> min); weather severity across the route (4.2 index); ground-time slack (−50 min).

## 10. Business insights

Full write-up in `reports/business_insights.md`.

**Ranking beats classifying.** The top two risk deciles hold **44%** of all delay
minutes:

| Decile | Delay rate | Lift | Share of delay minutes | Cumulative |
|---|---:|---:|---:|---:|
| D10 | 60.9% | 2.91× | 31.7% | 31.7% |
| D9 | 25.4% | 1.21× | 12.6% | 44.3% |
| D8 | 22.3% | 1.06× | 10.6% | 54.9% |
| ... | | | | |
| D1 | 8.0% | 0.38× | 3.6% | 100% |

**32% of all delay minutes are inherited**, not generated at the gate — which is
why a minute recovered before 09:00 is worth several recovered at 18:00.

**The threshold is an economics question.** With a $6,471 cost per delayed
departure, a $900 intervention recovering 30% of delay minutes, the break-even
probability is **0.46**, not 0.50:

| Policy | Net benefit (92-day test window) |
|---|---:|
| Intervene on every flight | −$25.95M |
| Heuristic: worse-than-median carrier, afternoon departure | −$1.85M |
| **Model at p ≥ 0.46** | **+$2.15M** |

**And the assumptions are stress-tested.** A 15-cell sensitivity grid over
intervention cost and effectiveness shows the optimal alert rate swinging from 2%
to 91% of flights, and net benefit turning **negative in 3 of 15 scenarios** — when
interventions are both expensive (≥$1,500) and weak (≤15% recovery). The binding
constraint is **intervention effectiveness, not model quality**. If a gate
intervention cannot recover ~20% of delay minutes, a better model will not save
the business case.

## 11. API

```bash
uvicorn api.main:app --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness, model type, feature-store vintage, live threshold |
| `GET /model-info` | hyperparameters, test metrics, feature list |
| `POST /predict` | one flight → probability, risk band, decision, SHAP reasons |
| `POST /delay-risk` | batch → ranked watch-list, flagged subset, expected count |

**Train/serve skew is handled explicitly.** Rolling features cannot be recomputed
for a single future flight, so `src/models/feature_store.py` freezes the latest
state of every aggregate into a 1 MB artifact (1,190 routes, 5,848
airport/weekday/hour demand cells, per-airport monthly weather normals). The
online featurizer degrades gracefully: missing weather falls back to
climatological normals, unknown tail numbers to carrier-typical airframes, absent
inbound status to a clean rotation with `prev_leg_known = 0`.

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "carrier_code":"MQ","origin":"ORD","destination":"LGA",
  "scheduled_departure_local":"2024-04-12T18:20:00",
  "prev_leg_arr_delay_min":65,"scheduled_ground_time_min":45,
  "origin_weather":{"precip_mm":9.0,"wind_gust_kt":41,"visibility_km":2.0,
                    "cloud_ceiling_ft":600,"condition":"Thunderstorm"}}'
# -> 0.961, band "high", recommend_intervention true
```

The same flight in clear weather with an on-time inbound scores 0.095.

## 12. Dashboard

```bash
streamlit run dashboard/app.py
```

Seven views: network overview, **live risk predictor** (scores a hypothetical
flight through the same code path as the API, with SHAP reasons), station league
table with the geographic map, carrier comparison including peak-hours-only,
weather impact, model performance, and an **economics view where moving the
intervention-cost and effectiveness sliders re-optimises the threshold and
recomputes net benefit live**.

---

## Repository layout

```
config/config.yaml           all tunable parameters; no magic numbers in code
src/data/                    generation, cleaning, integration
src/features/                engineering (6 families) + selection
src/analysis/                EDA figures and written summary
src/models/                  dataset/splits, training, tuning, SHAP,
                             feature store, online featurizer
src/insights/business.py     cost model, threshold optimisation, sensitivity
api/main.py                  FastAPI service
dashboard/app.py             Streamlit dashboard
tests/test_pipeline.py       24 tests, incl. 5 leakage guards
run_pipeline.py              orchestrator with --from / --to / --only
reports/                     figures, metrics, written analyses
```

## Testing

```
24 passed in 6.6s
```

Covering geometry, weather severity monotonicity, all cleaning defect classes,
time-zone alignment, cyclical encodings — and five leakage guards: unobservable
inbound arrivals, rotation-chain breaks, chronological split disjointness, and
target-derived columns in the feature set.

## Known limitations

1. **Flight data is simulated.** Structurally faithful and BTS-calibrated, but the
   absolute numbers describe a synthetic network. Swap in real BTS/NOAA data
   before quoting any figure externally.
2. **Weather uses observations, not forecasts.** Training attaches the observation
   valid at the scheduled hour. Production must attach the TAF forecast available
   at decision time, which will be less accurate and will lower measured
   performance. The join key and column names are identical, so the swap is one
   line in `attach_weather` — but do not skip it.
3. **Intervention cost and effectiveness are assumptions**, not measurements. The
   sensitivity grid quantifies exactly how much that matters; replace both with
   figures from a controlled rollout.
4. **Cancellations are excluded** from the modelling frame (a cancelled flight has
   no departure delay). Predicting cancellation is a related but distinct problem.
5. **No drift monitoring.** A production deployment needs feature-distribution and
   calibration monitoring; the feature store carries an `as_of` date as the hook
   for it.
