# Exploratory Data Analysis — Summary

Window **2023-01-01 → 2024-04-01**,
**264,958** operated flights, **35** airports,
**12** carriers, **1,190** routes.
System delay rate (≥15 min): **21.4%**.

## Shape of the problem
- 78.6% of flights depart within 15 minutes of schedule;
  0.6% are more than two hours late.
- When a flight is delayed, the median delay is 35 min
  and the mean is 44 min — the distribution is heavy-tailed,
  so averages understate the operational pain.

## Which airlines
- Best: **Delta Air Lines (14.4%)**; worst: **Envoy Air (29.9%)**.
- The spread between best and worst carrier is **15.6 percentage points**.
  Regional carriers are systematically worse, consistent with short turnarounds and
  deeper daily rotations.

## Which airports
- Largest single contributor to network delay minutes: **ORD (9.8% of network delay minutes)**.
- The top 3 origins account for **22.0%** of all delay minutes.
- Delay *rate* and delay *impact* rank differently: a small station can be chronically
  late without mattering to the network.

## When
- Worst month: **2024-04** (26.7%); best: **2024-03**.
- Worst weekday: **Fri** (23.9%); best: **Sat**.
- Delay risk compounds through the day: 15.4% at 06:00 vs
  **31.0% at 18:00**.
- Worst single slot in the week: **Fri 18:00 (37.2%)**.

## Weather
- Thunderstorms at the origin lift the delay rate to **48.3%**
  vs 21.3% otherwise — a **2.3x** multiplier.
- Low-IFR conditions (ceiling < 500 ft or visibility < 1.6 km): 31.3%.
- 6.2% of flights face non-trivial adverse weather; they
  account for an estimated **2.6%** of excess delays.

## Congestion and propagation
- Delay rate rises monotonically with the demand/capacity ratio at the origin.
- Propagation is the single strongest operational lever: with no upstream delay the rate
  is 17.3%; with 60–120 minutes of inherited delay it is
  **86.5%**.
- First leg of the day: 15.5% → deepest leg: **21.9%**.

## Holidays
- Flights inside a ±3-day major-holiday window run at 28.6%,
  **+40%** relative to normal days.

## Note on linear correlation
- Strongest single linear correlate of the target: upstream_delay_min (r=0.31).
  All individual correlations are modest, which is exactly why a non-linear model with
  interaction terms is warranted.
