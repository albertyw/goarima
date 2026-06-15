# Seasonal forecasting examples (SARIMAX)

These two worked examples show goarima's **seasonal differencing** in action and
compare it against statsmodels' **SARIMAX** (Seasonal ARIMA with eXogenous
regressors) reference. Both datasets are strongly seasonal monthly series, so
`AutoSARIMA` selects a seasonal difference (`D = 1`, `m = 12`) and the forecasts
reproduce the yearly cycle that a non-seasonal ARIMA would flatten.

## How these were generated

The orders are goarima's automatic choices, not hard-coded:

1. `example/main.go` runs `AutoSARIMA` on each series and prints a
   `[goarima-seasonal] <name> ARIMA(p,d,q)(P,D,Q)[m]` block with the forecast.
2. `example/plot_seasonal.py` parses those blocks, fits a statsmodels SARIMAX at
   the same `order` + `seasonal_order`, and plots each series' recent history with
   both forecasts continuing from the last observation (the dashed line).

Regenerate the charts (needs the `example/env` venv) with:

```sh
make charts            # runs plot_compare.py then plot_seasonal.py
# or directly:
cd example && env/bin/python plot_seasonal.py
```

Charts are written to the gitignored `example/charts/`; the committed copies below
live under `docs/images/`.

---

## AirPassengers — ARIMA(1,1,0)(0,1,0)[12]

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

---

## WineInd — ARIMA(2,1,1)(0,1,0)[12]

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
