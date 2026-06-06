package goarima

import (
	"math"

	"gonum.org/v1/gonum/optimize"
)

// refineCSS refines the Hannan-Rissanen seed (phiHR, thetaHR) by minimizing the
// conditional sum of squares (CSS) of the ARMA fit on the centered series z with
// a derivative-free Nelder-Mead search. Non-stationary / non-invertible
// parameter vectors are penalized with +Inf so the optimizer stays in the stable
// region.
//
// The refined coefficients are returned only if their CSS is strictly lower than
// the seed's (which, because the penalty is +Inf, also guarantees they are
// stationary and invertible); otherwise the seed is returned unchanged.
// Refinement therefore never returns a worse fit than the Hannan-Rissanen seed
// (Nelder-Mead finds a local optimum, not necessarily the global one) and never
// returns diverging coefficients.
func refineCSS(z, phiHR, thetaHR []float64) ([]float64, []float64) {
	p := len(phiHR)
	q := len(thetaHR)
	if p+q == 0 {
		return phiHR, thetaHR // nothing to optimize
	}

	cssAt := func(x []float64) float64 {
		phi := x[:p]
		theta := x[p:]
		if !isStationary(phi) || !isInvertible(theta) {
			return math.Inf(1)
		}
		var s float64
		for _, e := range armaResiduals(z, phi, theta) {
			s += e * e
		}
		return s
	}

	seed := make([]float64, p+q)
	copy(seed[:p], phiHR)
	copy(seed[p:], thetaHR)
	seedCSS := cssAt(seed)

	// Cap the function evaluations to bound the worst-case runtime on a
	// pathological (e.g. flat) CSS surface. The limit is far above what
	// Nelder-Mead needs to converge in p+q dimensions, and the best-so-far
	// result is still gated below, so a capped run stays correct.
	settings := &optimize.Settings{FuncEvaluations: 1000 * (p + q + 1)}
	result, err := optimize.Minimize(optimize.Problem{Func: cssAt}, seed, settings, &optimize.NelderMead{})
	if err != nil || result == nil {
		return phiHR, thetaHR
	}

	// Accept the optimum only if it strictly improves on the seed. A +Inf CSS
	// (non-stationary/non-invertible) can never satisfy this, so the returned
	// coefficients are always stable.
	if cssAt(result.X) < seedCSS {
		phi := make([]float64, p)
		copy(phi, result.X[:p])
		theta := make([]float64, q)
		copy(theta, result.X[p:])
		return phi, theta
	}
	return phiHR, thetaHR
}
