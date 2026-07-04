package goarima

import (
	"context"
	"errors"
	"fmt"
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
	ctx := cfg.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	if ctx.Err() != nil {
		return nil, fmt.Errorf("AutoARIMA: %w", ctx.Err())
	}

	dSeries := series
	if cfg.exog != nil {
		if _, err := validateExogMatrix(cfg.exog, len(series)); err != nil {
			return nil, fmt.Errorf("AutoARIMA: %w", err)
		}
		// Provisional level-scale β to net X out for differencing selection only;
		// each candidate Fit re-estimates β properly at the chosen d.
		if b0, err := olsBeta(series, cfg.exog); err == nil {
			dSeries = regressionResiduals(series, cfg.exog, b0)
		}
	}

	d := selectD(dSeries, maxD)
	space := searchSpace{
		series: series,
		d:      d,
		n:      len(series) - d, // differenced length, common to all (p,q)
		maxP:   maxP,
		maxQ:   maxQ,
		crit:   cfg.criterion,
		opts:   opts,
		ctx:    ctx,
	}

	var sel order
	if cfg.stepwise {
		sel = space.stepwiseSearch(cfg.parallel)
	} else {
		sel = space.gridSearch(cfg.parallel)
	}

	if ctx.Err() != nil {
		return nil, fmt.Errorf("AutoARIMA: %w", ctx.Err())
	}
	if sel[0] < 0 {
		return nil, errors.New("AutoARIMA: no candidate model could be fit")
	}

	best, err := NewARIMA(sel[0], d, sel[1])
	if err != nil {
		return nil, err
	}
	if err := best.Fit(series, opts...); err != nil {
		return nil, err
	}
	return best, nil
}

// AutoSARIMA selects seasonal ARIMA orders automatically for a known seasonal
// period m (m >= 2) and returns a fitted model. It chooses the seasonal
// differencing order D (0 or 1) with the seasonal-strength measure, then the
// regular differencing order d with the KPSS test on the seasonally-differenced
// series, then the non-seasonal orders p,q and the seasonal AR/MA orders P,Q with
// the same search AutoARIMA uses, over 0..maxP, 0..maxQ, 0..maxBigP, 0..maxBigQ
// (grid by default; WithStepwise / WithParallel honored).
//
// The criterion defaults to AIC (WithCriterion to change it). Any FitOption
// (WithCSSRefinement, WithMLE) is threaded through to every candidate fit and
// the final refit, exactly as in AutoARIMA.
func AutoSARIMA(series []float64, maxP, maxD, maxQ, maxBigP, maxBigQ, m int, opts ...FitOption) (*ARIMA, error) {
	if maxP < 0 || maxD < 0 || maxQ < 0 || maxBigP < 0 || maxBigQ < 0 {
		return nil, errors.New("AutoSARIMA: max orders must be non-negative")
	}
	if m < 2 {
		return nil, errors.New("AutoSARIMA: seasonal period m must be at least 2")
	}
	if len(series) < 2 {
		return nil, errors.New("AutoSARIMA: series too short")
	}
	if err := validateFinite(series); err != nil {
		return nil, fmt.Errorf("AutoSARIMA: %w", err)
	}

	var cfg fitConfig
	for _, opt := range opts {
		opt(&cfg)
	}
	ctx := cfg.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	if ctx.Err() != nil {
		return nil, fmt.Errorf("AutoSARIMA: %w", ctx.Err())
	}

	dSeries := series
	if cfg.exog != nil {
		if _, err := validateExogMatrix(cfg.exog, len(series)); err != nil {
			return nil, fmt.Errorf("AutoSARIMA: %w", err)
		}
		if b0, err := olsBeta(series, cfg.exog); err == nil {
			dSeries = regressionResiduals(series, cfg.exog, b0)
		}
	}

	bigD := selectSeasonalD(dSeries, m)
	s := SeasonalDifference(dSeries, m, bigD)
	d := selectD(s, maxD)

	space := searchSpace{
		series:  series,
		d:       d,
		n:       len(series) - bigD*m - d, // differenced length, common to all orders
		maxP:    maxP,
		maxQ:    maxQ,
		maxBigP: maxBigP,
		maxBigQ: maxBigQ,
		bigD:    bigD,
		period:  m,
		crit:    cfg.criterion,
		opts:    opts,
		ctx:     ctx,
	}

	var sel order
	if cfg.stepwise {
		sel = space.stepwiseSearch(cfg.parallel)
	} else {
		sel = space.gridSearch(cfg.parallel)
	}
	if ctx.Err() != nil {
		return nil, fmt.Errorf("AutoSARIMA: %w", ctx.Err())
	}
	if sel[0] < 0 {
		return nil, errors.New("AutoSARIMA: no candidate model could be fit")
	}

	best, err := NewSARIMA(sel[0], d, sel[1], sel[2], bigD, sel[3], m)
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
