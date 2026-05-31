# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`goarima` is a pure-Go implementation of ARIMA (AutoRegressive Integrated Moving Average) time-series modeling. It is a library (package `goarima`) with a separate `example/` command demonstrating usage. Linear systems are solved via the external `github.com/albertyw/gaussian` package.

## Commands

- `make test` — full check suite: installs lint deps, runs unit tests, `go vet`, `gofmt`, `go mod tidy`, `golangci-lint`, and `govulncheck`. This is the gate that must pass before committing.
- `make unit` — unit tests only with coverage written to `c.out`.
- `make race` — tests under the race detector.
- `make cover` — runs `make test` then prints per-function coverage.
- `make benchmark` — `go test -bench=. -benchmem`.
- Run a single test: `go test -run TestHannanRissanenARMA ./...`
- Run the demo: `cd example && go run .` (AutoARIMA + fixed-order fits on several classic datasets).
- `make example` — runs `example/compare.py`, which runs the Go demo (`go run .`) and statsmodels and prints their fixed-order results interleaved per dataset. Needs the `example/env` Python venv; falls back to goarima-only (`go run .`) if absent. Not part of `make test`/CI.

CI (`.drone.yml`) runs `make test`, `make race`, `make cover`, `make benchmark`, the profiling targets, and `checkmake` on the Makefile.

## Architecture

Two entry points: construct a model with explicit orders via `NewARIMA(p,d,q)` + `Fit`, or let `AutoARIMA(series, maxP, maxD, maxQ)` choose the orders and return a fitted model.

The `ARIMA.Fit` pipeline (`goarima.go`):

```
series ──Difference(d)──► y ──center (−mu)──► z (zero-mean, stationary)
   │                                            │
   └─ record anchors (last value of            └─► hannanRissanen(z,p,q)
      series differenced 0..d-1 times)              ├─ Stage 1: long AR(k) by Yule-Walker → ê
                                                     └─ Stage 2: OLS z_t ~ [z lags, ê lags] → phi, theta
                                            store lastY (p obs), lastE (q residuals), mu, anchors
```

Forecasting (`ARIMA.Forecast`) runs `forecastDiff` on the centered scale (rolling AR+MA recursion, future errors = 0), adds `mu` back, then integrates with `Undifference` once per differencing level using the stored `anchors` (a no-op when `d == 0`).

`AutoARIMA` (`autoarima.go`): `selectD` picks `d` by a variance heuristic (keep differencing while variance falls — note this over-differences positively-autocorrelated series); then a grid search over `p,q` minimizes `aic(n, sigma2, p, q)`, skipping `(0,0)` and any fit that errors.

Key files:
- `goarima.go` — `ARIMA` struct, `NewARIMA`, `Fit`, `Forecast`/`forecastDiff`, getters, `mean`/`meanSquare`/`isConstant`.
- `diff.go` — `Difference` / `Undifference`.
- `estimate.go` — `hannanRissanen` (the ARMA estimator), `hrAROrder`, `armaResiduals`.
- `yulewalker.go` — autocorrelation helpers and `solveYuleWalker` / `solveYuleWalkerFromAutocov` (used as the Hannan-Rissanen Stage 1 AR fit; guards constant series).
- `autoarima.go` — `AutoARIMA`, `selectD`, `aic`, `variance`.

Estimation is Hannan-Rissanen (pure linear algebra, no numerical optimizer). It is approximate — it does not match an MLE/CSS library (statsmodels/pmdarima) exactly. A CSS/MLE refinement is a possible future step.

The `ARIMA` struct fields are unexported; access state through the getter methods (`Orders`, `Phi`, `Theta`, `LastY`, `LastE`, `LastOrig`, `Sigma2`).

## Example data & reference scripts

- `example/main.go` runs `AutoARIMA` and fixed-order fits on several `example/data/*.csv` datasets (newline-separated values): airpassengers, lynx, wineind, sunspots, woolyrnq, austres. CSVs are exported from the vendored pmdarima `*.py` generators (or statsmodels for sunspots).
- `integration_test.go` embeds the AirPassengers CSV and asserts sensible (not reference-exact) behavior.
- `example/compare.py` is the comparison driver: it runs the Go example, fits `statsmodels` ARIMA at the same fixed orders as the Go `runFixed` examples, and prints the two interleaved per dataset (run via `make example`). It parses goarima's fixed-section blocks by dataset name, so the output format in `main.go` and the `FIXED` list in `compare.py` must stay in sync. statsmodels has no `auto_arima`, so the `AutoARIMA` section is goarima-only. Python deps are pinned in `example/pyproject.toml` and installed under `example/env`; reference-exact fixtures are not wired into the Go tests.

## Notes

- `docs/arima.md` is the beginner-friendly conceptual explainer (AR/MA/I, Yule-Walker, Hannan-Rissanen, AutoARIMA) with key equations and further-reading links; it cross-references the source files and is linked from the README. Keep it in sync if the algorithms change.
- Go 1.25; module path `github.com/albertyw/goarima`.
