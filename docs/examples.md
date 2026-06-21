# Forecasting examples

Worked examples of goarima on classic datasets, each compared against a
maximum-likelihood reference at the same orders:

- **ARIMA (non-seasonal)** — `AutoARIMA` vs [pmdarima](https://alkaline-ml.com/pmdarima/).
- **Prediction intervals** — `ForecastInterval` confidence bands vs pmdarima's.
- **Seasonal differencing (SARIMAX)** — `AutoSARIMA` vs statsmodels **SARIMAX**.

In every chart the grey line is the observed history, the dashed line marks the
last observation, and the two coloured lines are the forecasts continuing from it.
All orders are goarima's *automatic* choices, not hard-coded.

## How these were generated

1. `example/main.go` runs `AutoARIMA` (and `AutoSARIMA`) on each series and prints
   a `[goarima] <name> ARIMA(p,d,q)` block — and, for the seasonal datasets, a
   `[goarima-seasonal] <name> ARIMA(p,d,q)(P,D,Q)[m]` block — with the forecast.
2. `example/plot_compare.py`, `example/plot_interval.py`, and
   `example/plot_seasonal.py` parse those blocks, fit the reference (pmdarima /
   statsmodels SARIMAX) at the same orders, and plot each series' history with both
   forecasts (and, for `plot_interval.py`, the prediction bands).

Regenerate the charts (needs the `example/env` venv) with:

```sh
make charts            # runs plot_compare.py then plot_seasonal.py
```

Charts are written to the gitignored `example/charts/`; the committed copies below
live under `docs/images/`.

---

## ARIMA (non-seasonal)

`AutoARIMA` chooses `d` with a KPSS stationarity test, then searches `p` and `q`
to minimize an information criterion (AIC by default). Both goarima and pmdarima
are fitted with exact MLE here, so the AR/MA terms goarima picks let the two
forecasts follow each series' shape together.

```go
series, _ := /* example/data/<name>.csv */
model, _ := goarima.AutoARIMA(series, 5, 2, 5) // maxP, maxD, maxQ
forecast, _ := model.Forecast(horizon)
```

The orders `AutoARIMA` selected for each dataset (within `maxP=maxQ=5`, `maxD=2`):

| Dataset | Selected order | Horizon |
|---|---|---|
| AirPassengers | ARIMA(4,1,0) | 12 |
| Lynx | ARIMA(4,0,0) | 10 |
| WineInd | ARIMA(3,1,3) | 12 |
| Sunspots | ARIMA(4,1,4) | 10 |
| WoolyRnq | ARIMA(3,1,3) | 8 |
| AustRes | ARIMA(1,1,0) | 8 |

| AirPassengers — ARIMA(4,1,0) | Lynx — ARIMA(4,0,0) |
|---|---|
| ![AirPassengers AutoARIMA vs pmdarima](images/airpassengers.png) | ![Lynx AutoARIMA vs pmdarima](images/lynx.png) |

| Sunspots — ARIMA(4,1,4) | AustRes — ARIMA(1,1,0) |
|---|---|
| ![Sunspots AutoARIMA vs pmdarima](images/sunspots.png) | ![AustRes AutoARIMA vs pmdarima](images/austres.png) |

| WineInd — ARIMA(3,1,3) | WoolyRnq — ARIMA(3,1,3) |
|---|---|
| ![WineInd AutoARIMA vs pmdarima](images/wineind.png) | ![WoolyRnq AutoARIMA vs pmdarima](images/woolyrnq.png) |

Lynx (`d=0`) decays toward its mean, as a stationary AR forecast should. For the
seasonal series (AirPassengers, WineInd, WoolyRnq), a *non-seasonal* ARIMA only
partially captures the yearly cycle — which is exactly what seasonal differencing
fixes below.

---

## Prediction intervals

`ForecastInterval` returns the point forecast with a confidence band. The forecast
variance comes from the model's MA(∞) representation, `Var(k) = σ²·Σψ²`, so the
band widens with the horizon. The orders are the same `AutoARIMA` choices as above
(MLE-refit); the shaded band is goarima's 95% interval and the dashed lines are
[pmdarima](https://alkaline-ml.com/pmdarima/)'s at the same order.

```go
series, _ := /* example/data/<name>.csv */
model, _ := goarima.AutoARIMA(series, 5, 2, 5)
fc, _ := model.ForecastInterval(horizon, 0.95) // horizon, confidence level
// fc.Point, fc.Lower, fc.Upper, fc.StdErr
```

| AirPassengers — ARIMA(4,1,0), 95% | Lynx — ARIMA(4,0,0), 95% |
|---|---|
| ![AirPassengers prediction interval](images/airpassengers_interval.png) | ![Lynx prediction interval](images/lynx_interval.png) |

The two datasets show the two regimes. AirPassengers (`d=1`) is differenced, so the
band keeps fanning out as the horizon grows; the small vertical offset from
pmdarima is the same `d≥1` drift gap noted elsewhere — the interval *widths* still
match. Lynx (`d=0`) is stationary, so both the forecast and the band settle toward
constant levels, and goarima's band overlaps pmdarima's almost exactly.

---

## Seasonal differencing (SARIMAX)

These two strongly seasonal monthly series take a seasonal difference (`D = 1`,
`m = 12`), so the forecasts reproduce the yearly cycle that the non-seasonal fits
above flatten. The reference here is statsmodels' **SARIMAX** (Seasonal ARIMA with
eXogenous regressors).

### AirPassengers — ARIMA(1,1,0)(0,1,0)[12]

Monthly international airline passengers (1949–1960): a rising trend with a strong
12-month cycle that grows in amplitude.

![AirPassengers seasonal forecast: goarima AutoSARIMA vs statsmodels SARIMAX](images/airpassengers_seasonal.png)

**goarima:**

```go
series, _ := /* example/data/airpassengers.csv */
model, _ := goarima.AutoSARIMA(series, 3, 1, 3, 12) // maxP, maxD, maxQ, m
forecast, _ := model.Forecast(24)
// selected: ARIMA(1,1,0)(0,1,0)[12]
```

`AutoSARIMA` picked one ordinary difference (`d = 1`, KPSS) and one seasonal
difference (`D = 1`, the seasonal-strength test), then a single AR term.

**statsmodels SARIMAX reference** (the orange line):

```python
SARIMAX(series, order=(1, 1, 0), seasonal_order=(0, 1, 0, 12),
        enforce_stationarity=False, enforce_invertibility=False).fit()
```

The two forecasts are nearly identical — both follow the seasonal peaks and the
upward trend.

### WineInd — ARIMA(2,1,1)(0,1,0)[12]

Monthly Australian wine sales: a sharper, noisier 12-month cycle.

![WineInd seasonal forecast: goarima AutoSARIMA vs statsmodels SARIMAX](images/wineind_seasonal.png)

**goarima:**

```go
series, _ := /* example/data/wineind.csv */
model, _ := goarima.AutoSARIMA(series, 3, 1, 3, 12) // maxP, maxD, maxQ, m
forecast, _ := model.Forecast(24)
// selected: ARIMA(2,1,1)(0,1,0)[12]
```

Here the search added two AR terms and one MA term on top of the same
`(d=1, D=1, m=12)` differencing.

**statsmodels SARIMAX reference:**

```python
SARIMAX(series, order=(2, 1, 1), seasonal_order=(0, 1, 0, 12),
        enforce_stationarity=False, enforce_invertibility=False).fit()
```

Both forecasts track the seasonal swings closely; the small gaps come from the
different estimators (goarima's Hannan-Rissanen seed + drift vs SARIMAX's exact
likelihood) — the same drift difference documented in the integration tests.

---

> **Note on scope.** goarima implements seasonal *differencing*
> `(p, d, q)(0, D, 0)ₘ`. The multiplicative seasonal AR/MA polynomials (`P`, `Q`)
> are a later phase; see [`docs/arima.md`](arima.md) §7 for the details.
