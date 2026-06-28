package goarima

import (
	"math/cmplx"

	"gonum.org/v1/gonum/mat"
)

// Root-reflection repair for unstable Hannan-Rissanen estimates. The AR
// characteristic polynomial 1 − φ₁z − … − φ_pzᵖ (and the MA polynomial
// 1 + θ₁z + … + θ_qz^q) is stable/invertible iff all of its roots lie strictly
// outside the unit circle (see stability.go). When a fit lands inside, reflecting
// each offending root r to its reciprocal 1/conj(r) moves it outside while
// preserving the argument; conjugate pairs stay paired, so the rebuilt polynomial
// stays real. This is the classic Box-Jenkins repair, enabled by WithRootRepair.
const (
	rootTol    = 1e-7 // a root with |r| ≤ 1+rootTol is treated as on/inside the circle
	rootMargin = 1e-4 // reflected roots are pushed to at least this far outside
)

// polyRoots returns the roots of coef[0] + coef[1]·z + … + coef[k]·z^k via the
// eigenvalues of the companion matrix of the monic form. Trailing (high-order)
// zero coefficients are trimmed first; a constant (degree ≤ 0) has no roots.
func polyRoots(coef []float64) []complex128 {
	n := len(coef)
	for n > 0 && coef[n-1] == 0 {
		n--
	}
	if n <= 1 {
		return nil
	}
	deg := n - 1
	lead := coef[deg]
	a := make([]float64, deg) // monic: a[i] = coef[i]/lead, leading term 1
	for i := 0; i < deg; i++ {
		a[i] = coef[i] / lead
	}
	c := mat.NewDense(deg, deg, nil) // Frobenius companion matrix
	for i := 0; i < deg; i++ {
		c.Set(i, deg-1, -a[i])
	}
	for i := 1; i < deg; i++ {
		c.Set(i, i-1, 1)
	}
	var eig mat.Eigen
	eig.Factorize(c, mat.EigenKind(0)) // eigenvalues only, no vectors
	return eig.Values(nil)
}

// reflectUnstableRoots returns a polynomial (same coefficient convention as the
// input, constant term 1) whose roots are all strictly outside the unit circle,
// obtained by reflecting the offending roots of coef and rebuilding ∏(1 − z/rᵢ).
// A polynomial that is already stable is returned unchanged (a copy).
func reflectUnstableRoots(coef []float64) []float64 {
	roots := polyRoots(coef)
	if len(roots) == 0 {
		return append([]float64(nil), coef...)
	}
	changed := false
	for i, r := range roots {
		if cmplx.Abs(r) <= 1+rootTol {
			nr := complex(1, 0) / cmplx.Conj(r) // reflect inside -> outside
			if cmplx.Abs(nr) <= 1+rootMargin {  // unit root: nudge clear of the circle
				nr = nr / complex(cmplx.Abs(nr), 0) * complex(1+rootMargin, 0)
			}
			roots[i] = nr
			changed = true
		}
	}
	if !changed {
		return append([]float64(nil), coef...)
	}
	poly := []complex128{complex(1, 0)} // rebuild ∏(1 − z/rᵢ), constant term 1
	for _, r := range roots {
		inv := complex(1, 0) / r
		next := make([]complex128, len(poly)+1)
		for k := range poly {
			next[k] += poly[k]
			next[k+1] -= poly[k] * inv
		}
		poly = next
	}
	out := make([]float64, len(poly))
	for k := range poly {
		out[k] = real(poly[k]) // conjugate symmetry => imaginary parts cancel
	}
	return out
}

// repairStationary returns AR coefficients equivalent in degree to phi but with a
// stationary characteristic polynomial. Stationary input is returned unchanged.
func repairStationary(phi []float64) []float64 {
	if isStationary(phi) {
		return phi
	}
	coef := make([]float64, len(phi)+1)
	coef[0] = 1
	for i, v := range phi {
		coef[i+1] = -v
	}
	out := reflectUnstableRoots(coef)
	res := make([]float64, len(phi))
	for i := 1; i < len(out); i++ {
		res[i-1] = -out[i]
	}
	return res
}

// repairInvertible returns MA coefficients equivalent in degree to theta but with
// an invertible polynomial. Invertible input is returned unchanged.
func repairInvertible(theta []float64) []float64 {
	if isInvertible(theta) {
		return theta
	}
	coef := make([]float64, len(theta)+1)
	coef[0] = 1
	for i, v := range theta {
		coef[i+1] = v
	}
	out := reflectUnstableRoots(coef)
	res := make([]float64, len(theta))
	for i := 1; i < len(out); i++ {
		res[i-1] = out[i]
	}
	return res
}
