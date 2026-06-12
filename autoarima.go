package goarima

import (
	"errors"
	"fmt"
	"math"
)

// AutoARIMA selects ARIMA orders automatically and returns a fitted model.
//
// The differencing order d is chosen by the KPSS stationarity test (difference
// until the series tests stationary, up to maxD). The non-seasonal orders p
// and q are then chosen by an exhaustive grid search over 0..maxP and 0..maxQ
// that minimizes an information criterion; the (0,0) combination is skipped.
// Candidate orders whose fit fails (e.g. too few observations) are skipped.
//
// The criterion defaults to AIC and can be changed with WithCriterion (AIC,
// BIC, or AICc). Any FitOption (e.g. WithCSSRefinement, WithMLE) is threaded
// through to every candidate fit and the final refit, so candidates are scored
// and the final model is fit with the same options.
//
// Note that the criterion is always computed from the residual variance (see
// score), even when WithMLE is supplied: the refinement lowers each candidate's
// residual variance and thus influences selection, but the score is not the
// exact Gaussian-likelihood criterion. A likelihood-based criterion is left to
// a later phase.
func AutoARIMA(series []float64, maxP, maxD, maxQ int, opts ...FitOption) (*ARIMA, error) {
	if maxP < 0 || maxD < 0 || maxQ < 0 {
		return nil, errors.New("AutoARIMA: max orders must be non-negative")
	}
	if len(series) < 2 {
		return nil, errors.New("AutoARIMA: series too short")
	}
	if err := validateFinite(series); err != nil {
		return nil, fmt.Errorf("AutoARIMA: %w", err)
	}

	var cfg fitConfig
	for _, opt := range opts {
		opt(&cfg)
	}

	d := selectD(series, maxD)
	n := len(series) - d // length of the differenced series, common to all (p,q)

	bestScore := math.Inf(1)
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
			s := score(cfg.criterion, n, model.sigma2, p, q)
			if s < bestScore {
				bestScore = s
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
