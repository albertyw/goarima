package goarima

import (
	"math"

	"gonum.org/v1/gonum/optimize"
)

// refineCoefficients refines a Hannan-Rissanen seed (phiHR, thetaHR) by
// minimizing the objective fn over the stacked [phi, theta] parameter vector
// with a derivative-free Nelder-Mead search. fn receives the AR and MA parts
// separately and is expected to return +Inf for parameter vectors it wants to
// reject (e.g. non-stationary / non-invertible), keeping the optimizer in the
// stable region.
//
// The refined coefficients are returned only if their objective is strictly
// lower than the seed's; otherwise the seed is returned unchanged. Refinement
// therefore never returns a worse fit than the Hannan-Rissanen seed
// (Nelder-Mead finds a local optimum, not necessarily the global one) and, when
// fn rejects unstable parameters with +Inf, never returns diverging
// coefficients.
func refineCoefficients(phiHR, thetaHR []float64, fn func(phi, theta []float64) float64) ([]float64, []float64) {
	p := len(phiHR)
	q := len(thetaHR)
	if p+q == 0 {
		return phiHR, thetaHR // nothing to optimize
	}
	seed := make([]float64, p+q)
	copy(seed[:p], phiHR)
	copy(seed[p:], thetaHR)
	best := nelderMeadRefine(seed, func(x []float64) float64 { return fn(x[:p], x[p:]) })
	return copyFloats(best[:p]), copyFloats(best[p:])
}

// nelderMeadRefine minimizes objective starting from seed with a derivative-free
// Nelder-Mead search and returns the minimizer only if it strictly improves on
// the seed; otherwise it returns the seed unchanged. objective is expected to
// return +Inf for parameter vectors it wants to reject (e.g. non-stationary /
// non-invertible), keeping the optimizer in the stable region — such a vector can
// never beat the seed, so the result is always the seed or a strictly better
// stable point. The function-evaluation cap bounds the worst-case runtime on a
// pathological (e.g. flat) surface and is far above what Nelder-Mead needs.
func nelderMeadRefine(seed []float64, objective func([]float64) float64) []float64 {
	if len(seed) == 0 {
		return seed
	}
	seedObj := objective(seed)
	settings := &optimize.Settings{FuncEvaluations: 1000 * (len(seed) + 1)}
	result, err := optimize.Minimize(optimize.Problem{Func: objective}, seed, settings, &optimize.NelderMead{})
	if err != nil || result == nil {
		return seed
	}
	if objective(result.X) < seedObj {
		return result.X
	}
	return seed
}

// refineSeasonalCoefficients is the multiplicative-SARMA generalization of
// refineCoefficients: it refines the packed factor vector (φ, Φₛ, θ, Θₛ) by
// minimizing fn, which receives the four factors separately. Like the
// non-seasonal helper, the refined factors are returned only if they strictly
// improve on the seed.
func refineSeasonalCoefficients(phi, sphi, theta, stheta []float64, fn func(phi, sphi, theta, stheta []float64) float64) (rphi, rsphi, rtheta, rstheta []float64) {
	p, P, q := len(phi), len(sphi), len(theta)
	n := p + P + q + len(stheta)
	if n == 0 {
		return phi, sphi, theta, stheta
	}
	seed := make([]float64, n)
	copy(seed[:p], phi)
	copy(seed[p:p+P], sphi)
	copy(seed[p+P:p+P+q], theta)
	copy(seed[p+P+q:], stheta)
	best := nelderMeadRefine(seed, func(x []float64) float64 {
		return fn(x[:p], x[p:p+P], x[p+P:p+P+q], x[p+P+q:])
	})
	return copyFloats(best[:p]), copyFloats(best[p : p+P]), copyFloats(best[p+P : p+P+q]), copyFloats(best[p+P+q:])
}

// refineSeasonalCSS refines the multiplicative-SARMA seed by minimizing the
// conditional sum of squares of the expanded fit on the centered series z.
// Non-stationary / non-invertible factor vectors are penalized with +Inf.
func refineSeasonalCSS(z, phi, sphi, theta, stheta []float64, m int) (rphi, rsphi, rtheta, rstheta []float64) {
	return refineSeasonalCoefficients(phi, sphi, theta, stheta, func(phi, sphi, theta, stheta []float64) float64 {
		if !isStationary(phi) || !isStationary(sphi) || !isInvertible(theta) || !isInvertible(stheta) {
			return math.Inf(1)
		}
		var s float64
		for _, e := range armaResiduals(z, expandSeasonalAR(phi, sphi, m), expandSeasonalMA(theta, stheta, m)) {
			s += e * e
		}
		return s
	})
}

// refineCSS refines the Hannan-Rissanen seed (phiHR, thetaHR) by minimizing the
// conditional sum of squares (CSS) of the ARMA fit on the centered series z.
// Non-stationary / non-invertible parameter vectors are penalized with +Inf.
func refineCSS(z, phiHR, thetaHR []float64) ([]float64, []float64) {
	return refineCoefficients(phiHR, thetaHR, func(phi, theta []float64) float64 {
		if !isStationary(phi) || !isInvertible(theta) {
			return math.Inf(1)
		}
		var s float64
		for _, e := range armaResiduals(z, phi, theta) {
			s += e * e
		}
		return s
	})
}
