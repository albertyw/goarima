package goarima

import (
	"errors"
	"math"
)

// AutoARIMA selects ARIMA orders automatically and returns a fitted model.
//
// The differencing order d is chosen by a variance heuristic (keep differencing
// while the series variance decreases, up to maxD). The non-seasonal orders p
// and q are then chosen by an exhaustive grid search over 0..maxP and 0..maxQ
// that minimizes the Akaike Information Criterion (AIC); the (0,0) combination
// is skipped. Candidate orders whose fit fails (e.g. too few observations) are
// skipped.
func AutoARIMA(series []float64, maxP, maxD, maxQ int) (*ARIMA, error) {
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
			if err := model.Fit(series); err != nil {
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
	if err := best.Fit(series); err != nil {
		return nil, err
	}
	return best, nil
}

// selectD chooses the differencing order using a variance heuristic: it keeps
// differencing while the variance of the differenced series strictly decreases,
// up to maxD, and returns the order just before the variance stops falling.
func selectD(series []float64, maxD int) int {
	bestD := 0
	prevVar := variance(series)
	for d := 1; d <= maxD; d++ {
		diffed := Difference(series, d)
		if len(diffed) < 2 {
			break
		}
		v := variance(diffed)
		if v >= prevVar {
			break
		}
		prevVar = v
		bestD = d
	}
	return bestD
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

// variance returns the population variance of s, or 0 for an empty slice.
func variance(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	m := mean(s)
	var sum float64
	for _, v := range s {
		diff := v - m
		sum += diff * diff
	}
	return sum / float64(len(s))
}
