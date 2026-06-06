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

	objective := func(x []float64) float64 {
		return fn(x[:p], x[p:])
	}

	seed := make([]float64, p+q)
	copy(seed[:p], phiHR)
	copy(seed[p:], thetaHR)
	seedObj := objective(seed)

	// Cap the function evaluations to bound the worst-case runtime on a
	// pathological (e.g. flat) objective surface. The limit is far above what
	// Nelder-Mead needs to converge in p+q dimensions, and the best-so-far
	// result is still gated below, so a capped run stays correct.
	settings := &optimize.Settings{FuncEvaluations: 1000 * (p + q + 1)}
	result, err := optimize.Minimize(optimize.Problem{Func: objective}, seed, settings, &optimize.NelderMead{})
	if err != nil || result == nil {
		return phiHR, thetaHR
	}

	// Accept the optimum only if it strictly improves on the seed. A +Inf
	// objective (rejected by fn) can never satisfy this, so the returned
	// coefficients are always the stable seed or a strictly better stable point.
	if objective(result.X) < seedObj {
		phi := make([]float64, p)
		copy(phi, result.X[:p])
		theta := make([]float64, q)
		copy(theta, result.X[p:])
		return phi, theta
	}
	return phiHR, thetaHR
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
