package goarima

import "math"

// stable reports whether the polynomial
//
//	1 - a_1·z - a_2·z² - … - a_m·z^m
//
// has all of its roots strictly outside the unit circle. It uses the
// Schur-Cohn / reflection-coefficient (Levinson step-down) recursion: the
// polynomial is stable iff every reflection coefficient has magnitude < 1. An
// empty coefficient slice (degree 0) is trivially stable.
func stable(a []float64) bool {
	coef := append([]float64(nil), a...) // mutated in place by the step-down
	for m := len(coef); m >= 1; m-- {
		k := coef[m-1]
		if math.Abs(k) >= 1 {
			return false
		}
		denom := 1 - k*k
		next := make([]float64, m-1)
		for i := 0; i < m-1; i++ {
			next[i] = (coef[i] + k*coef[m-2-i]) / denom
		}
		coef = next
	}
	return true
}

// isStationary reports whether an AR model with coefficients phi (using the
// convention z_t = Σ phi_i·z_{t-i} + ε_t) is stationary — its characteristic
// polynomial 1 - Σ phi_i·z^i has all roots outside the unit circle.
func isStationary(phi []float64) bool {
	return stable(phi)
}

// isInvertible reports whether an MA model with coefficients theta (using the
// convention z_t = ε_t + Σ theta_j·ε_{t-j}) is invertible — its polynomial
// 1 + Σ theta_j·z^j has all roots outside the unit circle. That is the same
// test applied to the negated coefficients.
func isInvertible(theta []float64) bool {
	negated := make([]float64, len(theta))
	for i, v := range theta {
		negated[i] = -v
	}
	return stable(negated)
}
