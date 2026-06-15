# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`goarima` is a pure-Go implementation of ARIMA (AutoRegressive Integrated Moving Average) time-series modeling. It is a library (package `goarima`) with a separate `example/` command demonstrating usage. Linear systems are solved via the external `github.com/albertyw/gaussian` package; the optional CSS and exact-MLE refinements (`refine.go`, `mle.go`) use `gonum.org/v1/gonum/optimize` for the Nelder-Mead search.

## Commands

- `make test` — full check suite: installs lint deps, runs unit tests, `go vet`, `gofmt`, `go mod tidy`, `golangci-lint`, and `govulncheck`. This is the gate that must pass before committing.
- `make unit` — unit tests only with coverage written to `c.out`.
- `make race` — tests under the race detector.
- `make cover` — runs `make test` then prints per-function coverage.
- `make benchmark` — `go test -bench=. -benchmem`.
- Run a single test: `go test -run TestHannanRissanenARMA ./...`
- Run the demo: `cd example && go run .` (runs `AutoARIMA` on several classic datasets and prints the selected orders + forecasts).
- `make example` — runs `example/compare.py`, which runs the Go demo (`go run .`), parses the orders goarima's `AutoARIMA` selected, fits `pmdarima` at those same orders, and prints the two interleaved per dataset. Needs the `example/env` Python venv; falls back to goarima-only (`go run .`) if absent. Not part of `make test`/CI.
- `make charts` — runs `example/plot_compare.py` (reuses `compare.py`), which renders goarima-AutoARIMA-vs-pmdarima forecast charts at goarima's selected orders to the gitignored `example/charts/`. The committed copies linked from the README live under `docs/images/` (refresh by copying from `example/charts/`). Needs the `example/env` venv. Not part of `make test`/CI.

CI (`.drone.yml`) runs `make test`, `make race`, `make cover`, `make benchmark`, and `checkmake` on the Makefile.

## Architecture

Two entry points: construct a model with explicit orders via `NewARIMA(p,d,q)` + `Fit`, or let `AutoARIMA(series, maxP, maxD, maxQ)` choose the orders and return a fitted model. Seasonal *differencing* is available via `NewSARIMA(p,d,q,D,m)` and `AutoSARIMA(series, maxP, maxD, maxQ, m)` (seasonal AR/MA orders P,Q are a later phase — `SeasonalOrders()` returns them as 0). Both `Fit` and `AutoARIMA` are variadic in `FitOption`; `WithCSSRefinement()` enables the opt-in CSS refinement and `WithMLE()` the exact Gaussian MLE refinement (MLE takes precedence if both are passed). `Fit` and `AutoARIMA` reject a series containing NaN or ±Inf (`validateFinite`); `Forecast` errors until the model has been fitted (the `fitted` flag).

The `ARIMA.Fit` pipeline (`goarima.go`):

```
series ──Difference(d)──► y ──center (−mu)──► z (zero-mean, stationary)
   │                                            │
   └─ record anchors (last value of            └─► hannanRissanen(z,p,q)
      series differenced 0..d-1 times)              ├─ Stage 1: long AR(k) by Yule-Walker → ê
                                                     └─ Stage 2: OLS z_t ~ [z lags, ê lags] → phi, theta
                                            store lastY (p obs), lastE (q residuals), mu, anchors
```

With `WithCSSRefinement()` or `WithMLE()`, after Stage 2 the `(phi, theta)` seed is refined: `refineCSS` (`refine.go`) minimizes the conditional sum of squares, while `refineMLE` (`mle.go`) minimizes the exact Gaussian negative log-likelihood from a Kalman filter (`statespace.go`). Both run gonum Nelder-Mead through the shared `refineCoefficients` helper, penalizing non-stationary/non-invertible parameters with +Inf and falling back to the seed unless the refined fit strictly improves; residuals/`sigma2`/`lastE` are then recomputed from the refined coefficients. MLE takes precedence over CSS when both are requested.

Forecasting (`ARIMA.Forecast`) runs `forecastDiff` on the centered scale (rolling AR+MA recursion, future errors = 0), adds `mu` back, then integrates with `Undifference` once per differencing level using the stored `anchors` (a no-op when `d == 0`).

`AutoARIMA` (`autoarima.go` + `search.go`): `selectD` picks `d` with the KPSS level-stationarity test (`kpss.go`) — difference until the series tests stationary, up to maxD; then a search over `p,q` minimizes an information criterion (`score(crit, n, sigma2, p, q)` in `criterion.go`), skipping `(0,0)` and any fit that errors (non-stationary/non-invertible fits are rejected in `estimate.go`, see `stability.go`). The criterion defaults to AIC and is selectable with `WithCriterion(AIC|BIC|AICc)`. The search is an exhaustive grid by default; `WithStepwise()` switches to a Hyndman-Khandakar neighbor hill-climb (fewer fits, heuristic) and `WithParallel()` fits candidates concurrently (GOMAXPROCS workers, deterministic reduce — same result as serial, only faster when each fit is expensive, e.g. `WithMLE`). All three are `FitOption`s that only `AutoARIMA` reads; `Fit` ignores them.

Seasonal differencing (`seasonal.go`, `diff.go`): `NewSARIMA(p,d,q,D,m)` adds the `(1−Bᵐ)ᴰ` operator. `Fit` applies seasonal differencing *first* (recording the last `m` values at each level in `seasonalAnchors`), then the existing regular-diff/center/estimate pipeline runs unchanged on the seasonally-differenced series `s`; `Forecast` undoes regular differencing then seasonal differencing (`SeasonalUndifference`). `AutoSARIMA` picks `D` (0 or 1) with the Wang-Smith-Hyndman seasonal-strength measure `Fs` (`selectSeasonalD`/`seasonalStrength`/`centeredMovingAverage` in `seasonal.go`, threshold 0.64, matching R `forecast::nsdiffs(test="seas")`), then `d` via KPSS on `s`, then `p,q` via the same `searchSpace` (which carries `bigD`/`period`). All `AutoARIMA` `FitOption`s thread through `AutoSARIMA` identically.

Key files:
- `goarima.go` — `ARIMA` struct (incl. `bigD`/`period`/`seasonalAnchors`), `NewARIMA`, `NewSARIMA`, `Fit`, `Forecast`/`forecastDiff`, getters (`Orders`/`SeasonalOrders`/…), `mean`/`meanSquare`/`isConstant`.
- `diff.go` — `Difference` / `Undifference`, `SeasonalDifference` / `SeasonalUndifference`.
- `estimate.go` — `hannanRissanen` (the ARMA estimator), `hrAROrder`, `armaResiduals`.
- `refine.go` — `refineCoefficients` (the shared gonum Nelder-Mead helper) and `refineCSS`, the opt-in conditional-sum-of-squares refinement of the Hannan-Rissanen estimate.
- `mle.go` — `refineMLE`, the opt-in exact Gaussian maximum-likelihood refinement (Kalman-filter likelihood, gonum Nelder-Mead).
- `statespace.go` — `buildStateSpace` (Harvey ARMA state-space form), `solveLyapunov` (stationary init via the discrete Lyapunov equation), `kalmanConcentratedNLL` (Kalman prediction-error decomposition, concentrated negative log-likelihood).
- `stability.go` — `isStationary` / `isInvertible` (reflection-coefficient root test), used by the fit guards and the refinement penalty.
- `kpss.go` — `kpssLevelStationary`, the KPSS level-stationarity test used by `selectD`.
- `yulewalker.go` — autocovariance helpers and `solveYuleWalker` / `solveYuleWalkerFromAutocov` (used as the Hannan-Rissanen Stage 1 AR fit; guards constant series).
- `autoarima.go` — `AutoARIMA`, `AutoSARIMA`, `selectD` (order-search dispatch + final refit).
- `seasonal.go` — `selectSeasonalD` (seasonal-strength `Fs` test for D), `seasonalStrength`, `centeredMovingAverage`, `variance`.
- `criterion.go` — `Criterion` (AIC/BIC/AICc) and `score`, the information criterion AutoARIMA minimizes.
- `search.go` — `searchSpace` (carries `bigD`/`period` for seasonal fits), `evalCandidate`/`evalBatch` (optionally parallel), `gridSearch`, and `stepwiseSearch` (Hyndman-Khandakar).

By default, estimation is Hannan-Rissanen (pure linear algebra, no optimizer); `WithCSSRefinement()` adds a least-squares (CSS) refinement and `WithMLE()` an exact Gaussian maximum-likelihood refinement (Kalman filter, matching statsmodels' `method="statespace"`). The HR default and CSS refinement are approximate; MLE is the exact-likelihood fit, though small numeric differences from statsmodels remain.

The `ARIMA` struct fields are unexported; access state through the getter methods (`Orders`, `Phi`, `Theta`, `LastY`, `LastE`, `LastOrig`, `Sigma2`). The slice getters (`Phi`, `Theta`, `LastY`, `LastE`) return copies (`copyFloats`), so callers cannot mutate internal model state.

## Example data & reference scripts

- `example/main.go` runs `AutoARIMA` on several `example/data/*.csv` datasets (newline-separated values): airpassengers, lynx, wineind, sunspots, woolyrnq, austres. `runAuto` selects the order via `AutoARIMA` (fast HR path) then refits that order with `WithMLE()`, so the reported coefficients are maximum-likelihood — without that the Hannan-Rissanen seed shown would diverge from pmdarima's MLE for hard, weakly-identified orders (e.g. wineind's near-unit-root (3,1,3)). It prints one `[goarima] <name> ARIMA(p,d,q)` block per dataset (order + phi/theta/forecast); `compare.py` and `plot_compare.py` parse those blocks. It also prints one `[goarima-seasonal] AirPassengers SARIMA(p,d,q)(0,D,0)[m]` block from `AutoSARIMA` (distinct prefix so `compare.py`, which matches `[goarima]`, ignores it). CSVs are exported from the vendored pmdarima `*.py` generators (or statsmodels for sunspots).
- `integration_test.go` is the rigorous integration suite. It lives in the **external `goarima_test` package** so it exercises only the exported API. It embeds `example/data/*.csv` and the committed JSON fixtures and runs four tiers (no network/Python at test time): Tier 1a fixed-order fits vs pmdarima (`TestFixedOrdersMatchPmdarima` — coefficients only for pure AR/MA, forecasts only for `d==0`, because mixed-ARMA coefficients are non-identifiable and goarima adds a drift for `d>=1`); Tier 1b `auto_arima` selection vs pmdarima (`TestAutoSelectionVsPmdarima` — same `d`, p/q not required to match); Tier 2 analytic closed-forms via the public API; Tier 3 the goarima golden baseline (`TestGoldenWithMLE`). Seasonal differencing adds two more: `TestSeasonalFixedOrderMatchesPmdarima` (goarima `NewSARIMA` vs a statsmodels **SARIMAX** `seasonal_order=(0,D,0,m)` fit — AirPassengers `(1,1,0)(0,1,0)12`; phi only, since `d>=1` forecasts carry the drift gap) and `TestGoldenAutoSeasonalSelection` (an `AutoSARIMA` golden lock under the `auto_seasonal` key).
- `testdata/pmdarima_reference.json` — committed pmdarima fixtures (fixed-order + `auto_arima` + `seasonal_fixed`, the last a statsmodels/pmdarima **SARIMAX** `seasonal_order` fit), regenerated by `example/gen_reference.py` (`cd example && env/bin/python gen_reference.py`).
- `testdata/goarima_golden.json` — committed goarima `WithMLE` baseline (plus the `auto_seasonal` `AutoSARIMA` lock), regenerated by `go test -run TestGoldenWithMLE -update` (normal runs never write it).
- `example/compare.py` is the human eyeball comparison driver (run via `make example`): it runs the Go demo, parses the orders goarima's `AutoARIMA` chose, fits `pmdarima` at those same orders (pmdarima's default intercept adds a drift for `d>=1`, matching goarima), and prints the two interleaved per dataset. It depends on the `report` output format in `main.go` (the `[goarima] … ARIMA(p,d,q)` blocks), so keep that format and `parse_blocks` in sync.
- `example/plot_compare.py` (`make charts`) reuses `compare.py`'s helpers to draw the same goarima-AutoARIMA-vs-pmdarima comparison as charts.
- Order sync: the **demo** (`main.go`/`compare.py`/`plot_compare.py`) uses goarima's *auto-selected* orders, so nothing is hard-coded there. The **integration test** still uses *fixed* orders — those live in `example/gen_reference.py` and must stay in sync with the committed `testdata/*.json` fixtures (regenerate both when they change).
- Python deps are pinned in `example/pyproject.toml` and installed under `example/env`.

## Notes

- `docs/arima.md` is the beginner-friendly conceptual explainer (AR/MA/I, Yule-Walker, Hannan-Rissanen, AutoARIMA) with key equations and further-reading links; it cross-references the source files and is linked from the README. Keep it in sync if the algorithms change.
- Go 1.25; module path `github.com/albertyw/goarima`.
