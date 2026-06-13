# Understanding ARIMA (and how goarima implements it)

This document explains the ideas behind ARIMA time-series forecasting and the
specific algorithms `goarima` uses, for readers who are new to time series. It
leads with intuition, then gives the key equation for each piece and points to
the file in this repository where it lives. For a refresher or deeper dive, see
[Further reading](#further-reading).

Notation note: equations are written in plain text. `Σ_{i=1..p}` means "sum over
i from 1 to p", `y_{t-1}` means the value one step before time `t`, and `ε_t`
(epsilon) is random noise ("the part we cannot predict").

---

## 1. The forecasting problem

A *time series* is a sequence of measurements taken over time: monthly airline
passengers, yearly sunspot counts, daily prices. Forecasting asks: given the
values up to now (`y_1, y_2, …, y_n`), what are the next `h` values likely to be?

`goarima` produces **point forecasts** — a single best-guess number for each
future step — by fitting an ARIMA model to the history and rolling it forward.

---

## 2. Stationarity: the foundation

Most time-series math assumes the series is **stationary**: its statistical
behavior (mean, variance, how strongly consecutive points relate) does not drift
over time. A stationary series "looks the same" in any window — it wiggles around
a constant level.

Real data usually is *not* stationary: AirPassengers trends upward and grows in
amplitude. ARIMA's trick is to **transform** the data into something stationary,
model that, and then transform the forecasts back. The two transformations
goarima uses are **differencing** (Section 3, the "I") and **mean-centering**
(Section 4).

Why care? Because the estimation methods below (Yule-Walker, regression) only
give meaningful coefficients when the input is stationary.

---

## 3. The three letters: AR, I, MA

ARIMA(`p`, `d`, `q`) combines three ideas. `p`, `d`, and `q` are small
non-negative integers called the *orders*.

### AutoRegressive — AR(p)

**Intuition:** today is a weighted echo of recent days, plus noise. If sales were
high last month, they are probably high this month.

**Equation** (on a zero-mean series `z`):

```
z_t = φ_1·z_{t-1} + φ_2·z_{t-2} + … + φ_p·z_{t-p} + ε_t
```

- `φ_1 … φ_p` (phi) are the AR coefficients — how much each past value carries
  forward. There are `p` of them.
- `ε_t` is this step's unpredictable shock.

`p` is how many past *values* feed in.

### Integrated — I(d)

**Intuition:** a trending series is not stationary, but the *changes* from one
step to the next often are. "Integrated" refers to undoing differencing later.

**Differencing** replaces each value with the gap to the previous one:

```
∇y_t = y_t − y_{t-1}
```

Doing this `d` times (`d` is usually 0, 1, or 2) removes polynomial trends. `d=1`
turns a straight-line trend into a flat series; `d=2` handles curved trends.
goarima's `Difference` / `Undifference` live in `diff.go`.

### Moving Average — MA(q)

**Intuition:** today is also influenced by recent *surprises* (the noise terms),
not just recent values. A one-off shock can echo for a few steps.

**Equation:**

```
z_t = ε_t + θ_1·ε_{t-1} + θ_2·ε_{t-2} + … + θ_q·ε_{t-q}
```

- `θ_1 … θ_q` (theta) are the MA coefficients, weighting past shocks.
- `q` is how many past *errors* feed in.

(Confusingly, this "moving average" is unrelated to a rolling average of the
data — it is a weighted average of past *errors*.)

### Putting them together: ARIMA(p, d, q)

Difference the series `d` times to make it stationary, then model the result as
**ARMA(p, q)** — AR and MA combined:

```
z_t = Σ_{i=1..p} φ_i·z_{t-i}  +  ε_t  +  Σ_{j=1..q} θ_j·ε_{t-j}
```

where `z` is the differenced, mean-centered series. That single line is the heart
of ARIMA. Everything below is about (a) finding good `φ` and `θ` numbers, (b)
rolling the equation forward to forecast, and (c) picking `p`, `d`, `q`.

---

## 4. How goarima fits a model

`ARIMA.Fit` (in `goarima.go`) runs this pipeline:

```
series ──Difference(d)──► y ──center (−mean)──► z (zero-mean, stationary)
   │                                              │
   └─ remember "anchors" (last value of           └─► Hannan-Rissanen(z, p, q)
      the series differenced 0..d-1 times,             ├─ Stage 1: long AR(k) → noise proxies ê
      used to undo differencing later)                 └─ Stage 2: regress z on its lags + ê lags
                                            store φ, θ, mean, anchors, last values & residuals
```

### Step 1: difference and center

Difference `d` times (Section 3), then subtract the mean so the series is
centered on zero: `z_t = y_t − mean(y)`. Centering is how goarima represents the
model's constant/level term — the mean is stored and added back when forecasting.
A perfectly constant series is detected and yields zero coefficients (no division
by zero).

### Step 2: estimate AR with Yule-Walker

For a pure AR model, the coefficients satisfy the **Yule-Walker equations**, a
linear system built from the series' *autocovariances* `γ_k` (how the series
correlates with itself `k` steps apart):

```
R · φ = r

  R = Toeplitz matrix of γ_0 … γ_{p-1}    (a matrix where each descending
  r = [γ_1, γ_2, …, γ_p]                    diagonal is constant)
```

Solving this linear system gives `φ`. The leftover (white-noise) variance is
`σ² = γ_0 − Σ φ_i·γ_i`. goarima builds the autocovariances, assembles the
Toeplitz matrix, and solves the system in `yulewalker.go` (using the external
`gaussian` linear solver). This is exact, fast, and needs no iteration.

### Step 3: estimate ARMA with Hannan-Rissanen

The MA part is harder: the past errors `ε_{t-j}` are never observed directly.
The **Hannan-Rissanen** method (in `estimate.go`) sidesteps this with two
linear-regression stages — no numerical optimizer required:

1. **Stage 1 — approximate the noise.** Fit a deliberately *long* AR(`k`) model
   by Yule-Walker (goarima picks `k ≈ 2·ln n`). Its one-step residuals `ê_t`
   stand in for the unobservable errors `ε_t`.
2. **Stage 2 — one regression.** Ordinary least squares of `z_t` on its own past
   values *and* the proxy errors from Stage 1:

   ```
   z_t ≈ φ_1·z_{t-1} + … + φ_p·z_{t-p} + θ_1·ê_{t-1} + … + θ_q·ê_{t-q}
   ```

   The fitted regression weights are the AR coefficients `φ` and MA coefficients
   `θ`. (A pure-AR model, `q=0`, skips this and uses Yule-Walker directly.)

This is **approximate**: it is not maximum-likelihood, so the coefficients will
not exactly match libraries like statsmodels or pmdarima. The trade-off is
simplicity, determinism, and speed.

### Residual variance σ²

goarima recomputes the model's one-step residuals (`armaResiduals`) and sets
`σ²` to their mean square. `σ²` measures how well the model fits and feeds the
order-selection score in Section 6.

### Optional: CSS and exact-MLE refinement

Because Hannan-Rissanen is approximate, goarima offers two opt-in refinement
steps. Both treat the Hannan-Rissanen result as a starting guess and adjust
`φ`/`θ` with the same derivative-free Nelder-Mead search (the shared
`refineCoefficients` helper), differing only in the objective.

**CSS** (`Fit(series, WithCSSRefinement())`, in `refine.go`) minimizes the
**conditional sum of squares** — the sum of the squared one-step residuals:

```
minimize_{φ, θ}  Σ_t e_t²   where  e_t = z_t − Σ φ_i·z_{t-i} − Σ θ_j·e_{t-j}
```

**Exact MLE** (`Fit(series, WithMLE())`, in `mle.go`) minimizes the exact
Gaussian negative log-likelihood. It writes the ARMA model in state-space
(companion) form, initializes the state covariance from its stationary
distribution (the discrete **Lyapunov equation** `P = T·P·Tᵀ + R·Rᵀ`), and runs a
**Kalman filter** to get the one-step prediction errors `v_t` and their variances
`F_t`. Concentrating the innovation variance out (`σ̂² = (1/n)·Σ v_t²/F_t`) leaves
the objective

```
minimize_{φ, θ}  n·ln(σ̂²) + Σ_t ln(F_t)
```

This is the exact-likelihood fit that modern statsmodels uses
(`method="statespace"`), so it lands closer still than CSS — `statespace.go`
holds the state-space build, the Lyapunov solve, and the filter.

The same two safeguards keep both well-behaved: parameter sets outside the
stationary/invertible region (Section 7) are rejected during the search, and the
refined estimate is kept only if it strictly improves on the seed — otherwise the
Hannan-Rissanen seed is used unchanged. Refinement therefore never produces a
worse fit than the Hannan-Rissanen seed (Nelder-Mead finds a local optimum, not
necessarily the global one). If both options are passed, MLE takes precedence.

---

## 5. Forecasting

`ARIMA.Forecast` (in `goarima.go`) rolls the ARMA equation forward on the
centered scale, then reverses the transformations:

1. **Project ahead.** Apply `z_t = Σ φ_i·z_{t-i} + Σ θ_j·ε_{t-j}`, reusing the
   stored last values and residuals. Future errors are unknown, so they are set
   to their expected value, **zero**. (Because of this, the MA term only affects
   the first `q` steps; after that the forecast is driven by the AR part.)
2. **Add the mean back**, undoing the centering from Step 1.
3. **Integrate** (undo differencing) once per level of `d`, using the stored
   `anchors` — the last observed value at each differencing level. For `d=0` this
   is a no-op.

A stationary AR forecast decays toward the series mean; a differenced (`d≥1`)
forecast can keep trending, because integration accumulates the steps.

---

## 6. Choosing the orders automatically (AutoARIMA)

`AutoARIMA(series, maxP, maxD, maxQ)` (in `autoarima.go`) picks `p`, `d`, `q`
for you, in two stages.

### Choosing d (KPSS stationarity test)

Difference the series repeatedly (up to `maxD`) until it tests **stationary**,
and use that count as `d`. Stationarity is judged by the **KPSS test**
(Kwiatkowski-Phillips-Schmidt-Shin): it measures how far the running cumulative
sum of the demeaned series wanders relative to its long-run variance. A large
statistic means the level drifts (non-stationary) → difference and test again; a
small one means it is stable → stop. goarima uses the 5% level-stationarity
critical value (`kpss.go`).

This replaces an earlier variance heuristic that **over-differenced**
already-stationary but strongly autocorrelated series (e.g. it pushed sunspots
to `d=2`). The KPSS test keeps such series at `d=0` or `d=1`.

### Choosing p and q (information-criterion search)

With `d` fixed, try every `(p, q)` combination up to the limits (skipping the
empty `(0,0)` model), fit each, and keep the one with the lowest **information
criterion**. By default that is the **Akaike Information Criterion**:

```
AIC = n·ln(σ²) + 2·k        where k = p + q + 1
```

- `n·ln(σ²)` rewards a tight fit (small residual variance).
- `2·k` penalizes complexity (more coefficients), discouraging overfitting.

The model with the best balance of fit and simplicity wins. Because the criterion
is only comparable at a fixed `d` and sample size, the search runs after `d` is
chosen.

`WithCriterion` swaps in two alternatives (`criterion.go`):

```
BIC  = n·ln(σ²) + k·ln(n)              # heavier complexity penalty for n > ~7
AICc = AIC + 2k(k+1)/(n − k − 1)       # small-sample correction; +∞ once k ≥ n−1
```

**Stepwise search.** The default is an exhaustive grid. `WithStepwise()` instead
runs a **Hyndman-Khandakar** hill-climb (`search.go`): start from a few seed
orders, evaluate the neighbors `(p±1, q)`, `(p, q±1)`, `(p±1, q±1)`, move to the
best strictly-better one, and repeat until none improves. It fits far fewer models
(roughly linear in the path length instead of `O(p·q)`) but, being a heuristic,
can settle on a local rather than the global optimum. `WithParallel()` fits the
candidates of each step concurrently; the result is identical to the serial
search, so it only changes wall-clock time (and only helps when each fit is
expensive, e.g. under `WithMLE`).

---

## 7. What this implementation is *not*

goarima aims to be a clear, dependency-light, pure-Go ARIMA. It deliberately
leaves out things a production statistics package would include:

- **Approximate by default.** Hannan-Rissanen is approximate and the optional CSS
  refinement is least-squares. The optional `WithMLE` refinement adds the exact
  Gaussian (Kalman-filter) likelihood, but small numeric differences from
  statsmodels/pmdarima remain.
- **Unstable fits are rejected, not repaired.** If an explicit `(p,d,q)` lands
  outside the stationary/invertible region, `Fit` returns an error instead of
  re-estimating into the valid region.
- **No seasonal (SARIMA) terms**, and **point forecasts only** — no prediction
  intervals.

See the project README's *Limitations* section for the current list.

---

## 8. Notation glossary

| Symbol | Meaning |
|---|---|
| `y_t` | the observed series value at time `t` |
| `z_t` | the differenced, mean-centered (zero-mean) series |
| `p` | AR order — number of past *values* used |
| `d` | differencing order — how many times we take differences |
| `q` | MA order — number of past *errors* used |
| `φ_i` (phi) | AR coefficients |
| `θ_j` (theta) | MA coefficients |
| `ε_t` (epsilon) | white-noise error / shock at time `t` |
| `γ_k` (gamma) | autocovariance at lag `k` |
| `σ²` (sigma squared) | residual (white-noise) variance |
| `n` | number of observations |

---

## Further reading

Beginner-friendly:

- **Hyndman & Athanasopoulos, _Forecasting: Principles and Practice_ (3rd ed.),
  Chapter 9 — ARIMA models** — the best free, approachable introduction:
  <https://otexts.com/fpp3/arima.html>
- **Wikipedia — ARIMA:**
  <https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average>

Per-concept:

- **AR models & the Yule-Walker equations** (Wikipedia):
  <https://en.wikipedia.org/wiki/Autoregressive_model>
- **MA models** (Wikipedia):
  <https://en.wikipedia.org/wiki/Moving-average_model>
- **Akaike Information Criterion** (Wikipedia):
  <https://en.wikipedia.org/wiki/Akaike_information_criterion>

Reference / deeper:

- **Hannan-Rissanen estimation** — the original method: Hannan, E. J. &
  Rissanen, J. (1982), "Recursive estimation of mixed autoregressive-moving
  average order", _Biometrika_ 69(1), 81–94.
  <https://doi.org/10.1093/biomet/69.1.81>
- **statsmodels ARIMA** (the maximum-likelihood reference this project compares
  against):
  <https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html>
