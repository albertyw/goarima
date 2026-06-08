# goarima

A pure-Go implementation of ARIMA (AutoRegressive Integrated Moving Average)
time-series modeling, with automatic order selection.

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

`AutoARIMA` chooses `d` with a KPSS stationarity test and then grid-searches `p` and
`q` (up to the given maxima) to minimize the AIC, returning a fitted model.

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

### Inspecting a fitted model

```go
model.Orders()   // (p, d, q)
model.Phi()      // AR coefficients
model.Theta()    // MA coefficients
model.Sigma2()   // residual variance
```

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

The [`example/`](example/) directory contains a runnable demo that fits both
`AutoARIMA` and fixed-order models to several classic datasets (AirPassengers,
Lynx, wine sales, sunspots, wool production, and Australian population):

```sh
cd example && go run .
```

`make example` runs `example/compare.py`, which fits the same fixed-order
models with [statsmodels](https://www.statsmodels.org/) and prints the goarima
and statsmodels results interleaved per dataset for easy comparison. It requires
the Python environment described in `example/pyproject.toml` (installed under
`example/env`) and falls back to the goarima-only demo if that environment is
absent.

### Trend comparison

`make charts` renders goarima's exact-MLE forecast against
[pmdarima](https://alkaline-ml.com/pmdarima/)'s at the same fixed orders, writing
one chart per dataset to the gitignored `example/charts/`. Committed reference
copies live under [`docs/images/`](docs/images) (shown below). For stationary
(`d=0`) fits the two forecasts overlap; for differenced (`d≥1`) fits they can
separate slightly because goarima and pmdarima estimate the drift differently.

| Sunspots — ARIMA(2,0,1), forecasts overlap | AirPassengers — ARIMA(0,1,1), with drift |
|---|---|
| ![Sunspots forecast comparison](docs/images/sunspots.png) | ![AirPassengers forecast comparison](docs/images/airpassengers.png) |

## Limitations

This is an approximate, non-seasonal implementation. In particular:

- **Approximate by default.** The default Hannan-Rissanen fit is close to, but
  not identical to, statsmodels' default. The optional `WithMLE` refinement adds
  an exact Gaussian maximum-likelihood fit (Kalman filter), though small numeric
  differences from statsmodels remain.
- **Non-invertible/non-stationary fixed-order fits are rejected, not repaired.**
  `Fit` returns an error rather than estimating into the valid region, so some
  explicit `(p,d,q)` requests fail instead of producing a model.
- **No seasonal (SARIMA) terms** and **point forecasts only** (no prediction
  intervals).

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
selection against [pmdarima](https://alkaline-ml.com/pmdarima/), analytic
closed-forms, and a goarima golden baseline. Regenerate the fixtures (needs the
`example/env` venv) when goarima's numerics intentionally change:

```sh
cd example && env/bin/python gen_reference.py   # pmdarima reference fixtures
go test -run TestGoldenWithMLE -update          # goarima golden baseline
```

## License

[MIT](LICENSE)
