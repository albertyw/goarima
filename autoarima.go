package goarima

import (
	"errors"
	"math"
)

// AutoARIMA selects ARIMA orders automatically and returns a fitted model.
//
// The differencing order d is chosen by the KPSS stationarity test (difference
// until the series tests stationary, up to maxD). The non-seasonal orders p
// and q are then chosen by an exhaustive grid search over 0..maxP and 0..maxQ
// that minimizes the Akaike Information Criterion (AIC); the (0,0) combination
// is skipped. Candidate orders whose fit fails (e.g. too few observations) are
// skipped.
//
// Any FitOption (e.g. WithCSSRefinement) is threaded through to every candidate
// fit and the final refit, so candidates are scored and the final model is fit
// with the same options.
func AutoARIMA(series []float64, maxP, maxD, maxQ int, opts ...FitOption) (*ARIMA, error) {
	if maxP < 0 || maxD < 0 || maxQ < 0 {
		return nil, errors.New("AutoARIMA: max orders must be non-negative")
	}
	if len(series) < 2 {
		return nil, errors.New("AutoARIMA: series too short")
	}

	d := selectD(series, maxD)
	n := len(series) - d // length of the differenced series, common to all (p,q)

	bestAIC := math.Inf(1)
	bestP, bestQ := -1, -1
	for p := 0; p <= maxP; p++ {
		for q := 0; q <= maxQ; q++ {
			if p == 0 && q == 0 {
				continue
			}
			model, err := NewARIMA(p, d, q)
			if err != nil {
				continue
			}
			if err := model.Fit(series, opts...); err != nil {
				continue
			}
			a := aic(n, model.sigma2, p, q)
			if a < bestAIC {
				bestAIC = a
				bestP, bestQ = p, q
			}
		}
	}

	if bestP < 0 {
		return nil, errors.New("AutoARIMA: no candidate model could be fit")
	}

	best, err := NewARIMA(bestP, d, bestQ)
	if err != nil {
		return nil, err
	}
	if err := best.Fit(series, opts...); err != nil {
		return nil, err
	}
	return best, nil
}

// selectD chooses the differencing order with the KPSS stationarity test: it
// differences the series until it tests level-stationary, up to maxD, and
// returns that order. This avoids the over-differencing that a variance
// heuristic suffers on positively-autocorrelated (but already stationary) data.
func selectD(series []float64, maxD int) int {
	cur := series
	for d := 0; d < maxD; d++ {
		if len(cur) < 3 || kpssLevelStationary(cur) {
			return d
		}
		cur = Difference(cur, 1)
	}
	return maxD
}

// aic returns the Akaike Information Criterion for a model with the given
// residual variance and orders, using the Gaussian log-likelihood form
// n·ln(σ²) + 2k where k = p+q+1 (AR + MA coefficients plus the variance).
// A floor on σ² keeps the value finite for degenerate (perfectly fit) series,
// so ties are broken by the parameter count.
func aic(n int, sigma2 float64, p, q int) float64 {
	const floor = 1e-12
	if sigma2 < floor {
		sigma2 = floor
	}
	k := p + q + 1
	return float64(n)*math.Log(sigma2) + 2*float64(k)
}
