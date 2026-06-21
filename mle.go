package goarima

import "math"

// refineMLE refines the Hannan-Rissanen seed (phiHR, thetaHR) toward the exact
// Gaussian maximum-likelihood estimate by minimizing the concentrated negative
// log-likelihood computed with the Kalman filter (see statespace.go) on the
// centered series z. Non-stationary / non-invertible parameter vectors are
// penalized with +Inf so the optimizer stays in the stable region, and the
// refined estimate is kept only if it strictly improves on the HR seed (see
// refineCoefficients), so refinement never returns a worse or diverging fit.
//
// Unlike refineCSS, which minimizes the conditional sum of squares, this matches
// the exact-likelihood fit used by modern statsmodels (method="statespace").
func refineMLE(z, phiHR, thetaHR []float64) ([]float64, []float64) {
	return refineCoefficients(phiHR, thetaHR, func(phi, theta []float64) float64 {
		if !isStationary(phi) || !isInvertible(theta) {
			return math.Inf(1)
		}
		return kalmanConcentratedNLL(z, phi, theta)
	})
}

// refineSeasonalMLE is the multiplicative-SARMA generalization of refineMLE: it
// minimizes the exact Gaussian negative log-likelihood of the *expanded* model
// over the factor vector (φ, Φₛ, θ, Θₛ), so the fit matches statsmodels SARIMAX.
// Non-stationary / non-invertible factor vectors are penalized with +Inf.
func refineSeasonalMLE(z, phi, sphi, theta, stheta []float64, m int) (rphi, rsphi, rtheta, rstheta []float64) {
	return refineSeasonalCoefficients(phi, sphi, theta, stheta, func(phi, sphi, theta, stheta []float64) float64 {
		if !isStationary(phi) || !isStationary(sphi) || !isInvertible(theta) || !isInvertible(stheta) {
			return math.Inf(1)
		}
		return kalmanConcentratedNLL(z, expandSeasonalAR(phi, sphi, m), expandSeasonalMA(theta, stheta, m))
	})
}
