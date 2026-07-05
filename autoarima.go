package goarima

import (
	"context"
	"errors"
	"fmt"
)

// Bounds are the non-seasonal search bounds for AutoARIMA/AutoSARIMA: the
// maximum AR order (MaxP), differencing order (MaxD), and MA order (MaxQ) the
// search considers. Each is an inclusive upper bound on the range searched from
// 0.
type Bounds struct {
	MaxP int
	MaxD int
	MaxQ int
}

// SeasonalBounds are the seasonal search bounds for AutoSARIMA: the maximum
// seasonal AR order (MaxP) and seasonal MA order (MaxQ), plus the fixed seasonal
// Period (m >= 2). There is no seasonal MaxD — the seasonal differencing order D
// is auto-selected as 0 or 1 by the seasonal-strength test.
type SeasonalBounds struct {
	MaxP   int
	MaxQ   int
	Period int
}

// AutoARIMA selects ARIMA orders automatically and returns a fitted model.
//
// The differencing order d is chosen by the KPSS stationarity test (difference
// until the series tests stationary, up to max.MaxD). The non-seasonal orders p
// and q are then chosen by an exhaustive grid search over 0..max.MaxP and
// 0..max.MaxQ that minimizes an information criterion; the (0,0) combination is
// skipped. Candidate orders whose fit fails (e.g. too few observations) are
// skipped.
//
// The criterion defaults to AIC and can be changed with WithCriterion (AIC,
// BIC, or AICc). Any FitOption (e.g. WithMethod) is threaded through to every
// candidate fit and the final refit, so candidates are scored and the final
// model is fit with the same options.
//
// Note that the criterion is always computed from the residual variance (see
// score), even when WithMethod(MLE) is supplied: the refinement lowers each candidate's
// residual variance and thus influences selection, but the score is not the
// exact Gaussian-likelihood criterion. A likelihood-based criterion is left to
// a later phase.
func AutoARIMA(series []float64, max Bounds, opts ...AutoOption) (*ARIMA, error) {
	maxP, maxD, maxQ := max.MaxP, max.MaxD, max.MaxQ
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
		opt.applyAuto(&cfg)
	}
	fitOpts := fitOptions(opts)
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
		opts:   fitOpts,
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

	best, err := NewARIMA(Order{P: sel[0], D: d, Q: sel[1]})
	if err != nil {
		return nil, err
	}
	if err := best.Fit(series, fitOpts...); err != nil {
		return nil, err
	}
	return best, nil
}

// fitOptions returns the subset of opts that are also FitOptions — the
// fit-relevant options (WithMethod, WithExog, WithRootRepair) — to thread through
// to every candidate fit and the final refit. Search-only options
// (WithCriterion, WithStepwise, WithParallel, WithContext) implement only
// AutoOption and are excluded.
func fitOptions(opts []AutoOption) []FitOption {
	var out []FitOption
	for _, opt := range opts {
		if fo, ok := opt.(FitOption); ok {
			out = append(out, fo)
		}
	}
	return out
}

// AutoSARIMA selects seasonal ARIMA orders automatically for a known seasonal
// period seasonal.Period (>= 2) and returns a fitted model. It chooses the
// seasonal differencing order D (0 or 1) with the seasonal-strength measure, then
// the regular differencing order d with the KPSS test on the seasonally-
// differenced series, then the non-seasonal orders p,q and the seasonal AR/MA
// orders P,Q with the same search AutoARIMA uses, over 0..max.MaxP, 0..max.MaxQ,
// 0..seasonal.MaxP, 0..seasonal.MaxQ (grid by default; WithStepwise /
// WithParallel honored).
//
// The criterion defaults to AIC (WithCriterion to change it). Any FitOption
// (e.g. WithMethod) is threaded through to every candidate fit and the final
// refit, exactly as in AutoARIMA.
func AutoSARIMA(series []float64, max Bounds, seasonal SeasonalBounds, opts ...AutoOption) (*ARIMA, error) {
	maxP, maxD, maxQ := max.MaxP, max.MaxD, max.MaxQ
	maxBigP, maxBigQ, m := seasonal.MaxP, seasonal.MaxQ, seasonal.Period
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
		opt.applyAuto(&cfg)
	}
	fitOpts := fitOptions(opts)
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
		opts:    fitOpts,
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

	best, err := NewSARIMA(Order{P: sel[0], D: d, Q: sel[1]}, SeasonalOrder{P: sel[2], D: bigD, Q: sel[3], Period: m})
	if err != nil {
		return nil, err
	}
	if err := best.Fit(series, fitOpts...); err != nil {
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
