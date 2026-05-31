# goarima

A pure-Go implementation of ARIMA (AutoRegressive Integrated Moving Average)
time-series modeling, with automatic order selection.

It fits and forecasts ARIMA(p, d, q) models using only linear algebra — no CGo
and no numerical optimizer. Coefficients are estimated with the Hannan-Rissanen
method, so fitting is deterministic and fast. The trade-off is that estimates
are approximate: they will not match a maximum-likelihood library
(statsmodels / pmdarima) exactly. See [Limitations](#limitations).

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

`AutoARIMA` chooses `d` by a variance heuristic and then grid-searches `p` and
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

## Limitations

This is an approximate, non-seasonal implementation. In particular:

- **Hannan-Rissanen is not MLE.** Coefficients are close to, but not identical
  to, statsmodels/pmdarima.
- **`d` selection over-differences positively-autocorrelated series.** The
  variance heuristic is simpler than a KPSS/ADF stationarity test.
- **No stationarity/invertibility enforcement.** A fitted model with a
  non-invertible MA term can produce a diverging forecast.
- **No seasonal (SARIMA) terms** and **point forecasts only** (no prediction
  intervals).

## Development

```sh
make test      # unit tests, vet, gofmt, go mod tidy, golangci-lint, govulncheck
make race      # race detector
make cover     # coverage report
make benchmark # benchmarks
```

## License

[MIT](LICENSE)
