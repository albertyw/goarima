# goarima

[![Build Status](https://drone.albertyw.com/api/badges/albertyw/goarima/status.svg)](https://drone.albertyw.com/albertyw/goarima)
[![Go Reference](https://pkg.go.dev/badge/github.com/albertyw/goarima.svg)](https://pkg.go.dev/github.com/albertyw/goarima)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Sunspots forecast comparison](docs/images/sunspots.png)

A pure-Go implementation of ARIMA (AutoRegressive Integrated Moving Average)
time-series modeling, with automatic order selection, prediction intervals,
full multiplicative seasonal `(p,d,q)(P,D,Q)ₘ` models, and exogenous regressors
(the full SARIMAX family, validated against statsmodels SARIMAX).

It fits and forecasts ARIMA(p, d, q) models. By default, coefficients are
estimated with the Hannan-Rissanen method (pure linear algebra), so fitting is
deterministic and fast. The trade-off is that the estimates are approximate:
they will not match a maximum-likelihood library (statsmodels / pmdarima)
exactly. Two optional refinement steps tighten the coefficients: a
conditional-sum-of-squares search (`WithCSSRefinement`) and an exact Gaussian
maximum-likelihood fit via the Kalman filter (`WithMLE`, matching statsmodels'
`method="statespace"` default). See [Limitations](#limitations).

**New to time series?** [`docs/arima.md`](docs/arima.md) explains ARIMA and every
algorithm implemented here — in plain language with the key equations — for
readers new to time-series forecasting.

## Install

```sh
go get github.com/albertyw/goarima
```

Requires Go 1.25+.

## Usage

### Automatic order selection

`AutoARIMA` chooses `d` with a KPSS stationarity test and then searches `p` and
`q` (up to the given maxima) to minimize an information criterion, returning a
fitted model. By default it minimizes the AIC with an exhaustive grid search; see
[Order-search options](#order-search-options) to change the criterion or strategy.

```go
package main

import (
	"fmt"

	"github.com/albertyw/goarima"
)

func main() {
	series := []float64{112, 118, 132, 129, 121, 135, 148, 148, 136, 119 /* … */}

	model, err := goarima.AutoARIMA(series, 5, 2, 5) // maxP, maxD, maxQ
	if err != nil {
		panic(err)
	}

	p, d, q := model.Orders()
	fmt.Printf("selected ARIMA(%d,%d,%d)\n", p, d, q)

	forecast, err := model.Forecast(12) // next 12 steps
	if err != nil {
		panic(err)
	}
	fmt.Println(forecast)
}
```

### Fixed orders

When you already know the orders, construct the model directly:

```go
model, err := goarima.NewARIMA(1, 1, 1) // p, d, q
if err != nil {
	panic(err)
}
if err := model.Fit(series); err != nil {
	panic(err)
}
forecast, err := model.Forecast(10)
```

`Fit` returns an error for a series that is too short or contains a NaN or
infinite value, and `Forecast` returns an error if the model has not been fitted
yet.

### Seasonal models (SARIMA)

For seasonal data with a known period `m`, goarima fits the full multiplicative
SARIMA `(p,d,q)(P,D,Q)ₘ` model — seasonal differencing `(1−Bᵐ)ᴰ` plus seasonal
AR/MA factors `Φₛ(Bᵐ)`/`Θₛ(Bᵐ)`. `NewSARIMA` takes the seasonal orders `P, D, Q`
and period `m`; `AutoSARIMA` selects `D` (via the seasonal-strength test), then
`d`, `p`, `q`, `P`, and `Q` automatically:

```go
// Explicit: the airline model ARIMA(0,1,1)(0,1,1) with period 12.
model, err := goarima.NewSARIMA(0, 1, 1, 0, 1, 1, 12) // p, d, q, P, D, Q, m

// Automatic: choose p, d, q, P, Q (and D) for a monthly (m=12) series.
model, err := goarima.AutoSARIMA(series, 3, 1, 3, 1, 1, 12) // maxP, maxD, maxQ, maxBigP, maxBigQ, m
```

`SeasonalOrders()` returns `(P, D, Q, m)`, and `SeasonalPhi()`/`SeasonalTheta()`
return the seasonal AR/MA factors (`Phi()`/`Theta()` return the regular ones).
The fit is validated against statsmodels'
[SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
class at `seasonal_order=(P,D,Q,m)`. `AutoSARIMA` accepts the same options as
`AutoARIMA`.

### Exogenous regressors (SARIMAX)

To let external predictors drive the series, pass an `n×k` regressor matrix `X`
with `WithExog`. goarima fits *regression with ARIMA errors*,
`yₜ = Xₜ·β + ηₜ`, where `ηₜ` follows the ARIMA model — the same parameterization
as statsmodels SARIMAX with `exog=` and `trend="n"`. Read the coefficients with
`Beta()` and forecast with `ForecastExog` / `ForecastIntervalExog`, supplying the
future regressor rows (`h×k`):

```go
model, err := goarima.NewARIMA(1, 0, 0)
err = model.Fit(series, goarima.WithExog(X), goarima.WithMLE())
beta := model.Beta()                          // estimated β (length k)
forecast, err := model.ForecastExog(12, futureX) // futureX is 12×k
fc, err := model.ForecastIntervalExog(12, 0.95, futureX)
```

`β` is estimated jointly with the ARMA coefficients under `WithMLE` /
`WithCSSRefinement` (a two-step OLS seed otherwise). `WithExog` also works with
`NewSARIMA` and threads through `AutoARIMA` / `AutoSARIMA`. A model fit with
exogenous regressors must use `ForecastExog` rather than `Forecast` (and
vice-versa); calling the wrong one returns an error.

### Coefficient refinement

By default `Fit` uses the Hannan-Rissanen estimate. Two opt-in options refine it
with a derivative-free Nelder-Mead search seeded from the Hannan-Rissanen fit:

- `WithCSSRefinement()` minimizes the conditional sum of squares (a least-squares
  fit).
- `WithMLE()` minimizes the exact Gaussian negative log-likelihood computed with
  a Kalman filter, matching statsmodels' `method="statespace"` default.

Both move the coefficients toward a maximum-likelihood fit and never make the fit
worse — a refined estimate is kept only if it is stationary, invertible, and
strictly improves on the seed, otherwise the seed is used unchanged. If both
options are supplied, `WithMLE` takes precedence.

```go
err := model.Fit(series, goarima.WithMLE())
// AutoARIMA accepts the same options and threads them through every candidate fit:
model, err := goarima.AutoARIMA(series, 5, 2, 5, goarima.WithMLE())
```

### Order-search options

`AutoARIMA` takes three further options that tune *how* the orders are searched
(they only affect `AutoARIMA`; `Fit` ignores them):

- `WithCriterion(c)` — the information criterion to minimize: `AIC` (default),
  `BIC` (penalizes extra parameters more), or `AICc` (small-sample-corrected AIC).
- `WithStepwise()` — replace the exhaustive grid with a Hyndman-Khandakar stepwise
  neighbor search. It fits far fewer candidates (a hill-climb from a few seed
  orders) but is a heuristic and can miss the grid's global optimum.
- `WithParallel()` — fit candidate orders concurrently across `GOMAXPROCS`
  goroutines. Selection is deterministic and identical to the serial search, so it
  only changes speed — and it pays off only when each fit is expensive (e.g. with
  `WithMLE`); for the fast default Hannan-Rissanen fits the goroutine overhead
  outweighs the benefit.

```go
model, err := goarima.AutoARIMA(series, 5, 2, 5,
    goarima.WithCriterion(goarima.BIC),
    goarima.WithStepwise(),
)
```

### Prediction intervals

`ForecastInterval` returns the point forecast together with a two-sided
prediction interval at a given confidence level. The standard errors come from
the model's MA(∞) representation (`Var(k steps) = σ²·Σψ²`), with the differencing
operators folded into the AR side so the bounds are on the original scale:

```go
fc, err := model.ForecastInterval(12, 0.95) // horizon, confidence level
if err != nil {
	panic(err)
}
fc.Point  // point forecast (identical to Forecast(12))
fc.Lower  // lower bounds
fc.Upper  // upper bounds
fc.StdErr // forecast standard errors
```

The interval widths match statsmodels' `get_forecast().conf_int()`.

### Inspecting a fitted model

```go
model.Orders()   // (p, d, q)
model.Phi()      // AR coefficients (copy)
model.Theta()    // MA coefficients (copy)
model.Beta()     // exogenous regression coefficients (copy; empty without WithExog)
model.Sigma2()   // residual variance
```

The slice getters return copies, so mutating the result never affects the model.

The package also exposes the `Difference` / `Undifference` helpers used
internally.

## How it works

```
series ──Difference(d)──► center (−mean) ──► Hannan-Rissanen ──► phi, theta
                                                  │
                                  Stage 1: long AR(k) by Yule-Walker → residual proxies
                                  Stage 2: OLS of the series on its lags + residual-proxy lags
```

`Forecast` runs the AR+MA recursion forward (future errors = 0), adds the mean
back, and integrates once per differencing level to return values on the
original scale.

For a full, beginner-friendly walkthrough of these algorithms — AR/MA/I,
Yule-Walker, Hannan-Rissanen, and AutoARIMA's order selection — with the key
equations and links for further reading, see [`docs/arima.md`](docs/arima.md).

## Examples

The [`example/`](example/) directory contains a runnable demo that runs
`AutoARIMA` on several classic datasets (AirPassengers, Lynx, wine sales,
sunspots, wool production, and Australian population) and prints the selected
orders, coefficients, and forecasts:

```sh
cd example && go run .
```

`make example` runs `example/compare.py`, which fits
[pmdarima](https://alkaline-ml.com/pmdarima/) at the **orders goarima's
AutoARIMA selected** for each dataset and prints the two results interleaved for
easy comparison. It requires the Python environment described in
`example/pyproject.toml` (installed under `example/env`) and falls back to the
goarima-only demo if that environment is absent.

### Trend comparison

`make charts` renders goarima's AutoARIMA forecast against
[pmdarima](https://alkaline-ml.com/pmdarima/)'s at the same goarima-selected
order, writing one chart per dataset to the gitignored `example/charts/`.
Committed copies live under [`docs/images/`](docs/images) (two shown below). Both
sides are exact-MLE fits, so the AR terms goarima picks let the two forecasts
follow each series' cyclic shape together (for over-parameterized orders such as
wineind's near-unit-root (3,1,3) the amplitudes can still differ).

| Wool production (woolyrnq) — goarima AutoARIMA vs pmdarima | AirPassengers — goarima AutoARIMA vs pmdarima |
|---|---|
| ![Wool Production forecast comparison](docs/images/woolyrnq.png) | ![AirPassengers forecast comparison](docs/images/airpassengers.png) |

### Seasonal forecasting (SARIMAX)

[`docs/examples.md`](docs/examples.md) collects worked examples — non-seasonal
`AutoARIMA` (vs pmdarima) and the two seasonal examples below, AirPassengers and
WineInd, comparing `AutoSARIMA` against statsmodels **SARIMAX** — with the charts
inline and the exact settings used to generate them (`example/plot_seasonal.py`).

| AirPassengers — goarima AutoSARIMA vs statsmodels SARIMAX | WineInd — goarima AutoSARIMA vs statsmodels SARIMAX |
|---|---|
| ![AirPassengers seasonal forecast](docs/images/airpassengers_seasonal.png) | ![WineInd seasonal forecast](docs/images/wineind_seasonal.png) |

### Regression with ARIMA errors (SARIMAX exog)

[`docs/examples.md`](docs/examples.md) also walks through an exogenous-regressor
example (`example/plot_exog.py`): a covariate-driven demand series where
`WithExog` lets the `ForecastExog` forecast track a known future covariate while a
plain ARIMA reverts to the mean. goarima's exog forecast lands on top of
statsmodels SARIMAX at the same order.

![Regression with ARIMA errors](docs/images/exog.png)

## Limitations

This is a pure-Go implementation that aims for clarity over completeness:

- **Approximate by default.** The default Hannan-Rissanen fit is close to, but
  not identical to, statsmodels' default. The optional `WithMLE` refinement adds
  an exact Gaussian maximum-likelihood fit (Kalman filter), though small numeric
  differences from statsmodels remain. For seasonal AR/MA models, the fast default
  uses an approximate seasonal seed; `WithMLE` matches SARIMAX.
- **Non-invertible/non-stationary fixed-order fits are rejected, not repaired.**
  `Fit` returns an error rather than estimating into the valid region, so some
  explicit `(p,d,q)` requests fail instead of producing a model.

## Development

```sh
make test      # unit tests, vet, gofmt, go mod tidy, golangci-lint, govulncheck
make race      # race detector
make cover     # coverage report
make benchmark # benchmarks
make charts    # trend-comparison charts -> example/charts/ (gitignored; needs example/env)
```

### Integration tests

`integration_test.go` (in the external `goarima_test` package, so it uses only
the exported API) compares goarima against committed reference fixtures — no
network or Python at test time. It checks fixed-order fits and `auto_arima`
selection against [pmdarima](https://alkaline-ml.com/pmdarima/), a seasonal
`NewSARIMA` fit against statsmodels
[SARIMAX](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html),
analytic closed-forms, and a goarima golden baseline (including an `AutoSARIMA`
lock). Regenerate the fixtures (needs the `example/env` venv) when goarima's
numerics intentionally change:

```sh
cd example && env/bin/python gen_reference.py   # pmdarima reference fixtures
go test -run TestGoldenWithMLE -update          # goarima golden baseline
```

## License

[MIT](LICENSE)
