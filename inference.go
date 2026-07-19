package goarima

import (
	"errors"
	"fmt"
	"math"

	"github.com/albertyw/gaussian"
)

// numHessian returns the symmetric Hessian of f at x, approximated by central
// second-order finite differences. The per-coordinate step is scaled to each
// argument's magnitude so it works across parameter scales.
func numHessian(f func([]float64) float64, x []float64) [][]float64 {
	n := len(x)
	h := make([]float64, n)
	for i, xi := range x {
		h[i] = 1e-4 * math.Max(1, math.Abs(xi))
	}
	step := func(deltas map[int]float64) []float64 {
		y := append([]float64(nil), x...)
		for i, s := range deltas {
			y[i] += s
		}
		return y
	}
	f0 := f(x)
	H := make([][]float64, n)
	for i := range H {
		H[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		fp := f(step(map[int]float64{i: h[i]}))
		fm := f(step(map[int]float64{i: -h[i]}))
		H[i][i] = (fp - 2*f0 + fm) / (h[i] * h[i])
		for j := i + 1; j < n; j++ {
			fpp := f(step(map[int]float64{i: h[i], j: h[j]}))
			fpm := f(step(map[int]float64{i: h[i], j: -h[j]}))
			fmp := f(step(map[int]float64{i: -h[i], j: h[j]}))
			fmm := f(step(map[int]float64{i: -h[i], j: -h[j]}))
			H[i][j] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
			H[j][i] = H[i][j]
		}
	}
	return H
}

// packCoef stacks the fitted coefficients into the canonical order
// (β, φ, Φₛ, θ, Θₛ) used by paramNames, StdErrors, and the NLL objectives.
func packCoef(beta, phi, sphi, theta, stheta []float64) []float64 {
	out := make([]float64, 0, len(beta)+len(phi)+len(sphi)+len(theta)+len(stheta))
	out = append(out, beta...)
	out = append(out, phi...)
	out = append(out, sphi...)
	out = append(out, theta...)
	out = append(out, stheta...)
	return out
}

// paramNames returns the canonical coefficient names (excluding sigma2) in the
// order (β, φ, Φₛ, θ, Θₛ), mirroring statsmodels' labels.
func paramNames(exogDim, p, bigP, q, bigQ, period int) []string {
	var names []string
	for i := 1; i <= exogDim; i++ {
		names = append(names, fmt.Sprintf("x%d", i))
	}
	for i := 1; i <= p; i++ {
		names = append(names, fmt.Sprintf("ar.L%d", i))
	}
	for i := 1; i <= bigP; i++ {
		names = append(names, fmt.Sprintf("ar.S.L%d", i*period))
	}
	for i := 1; i <= q; i++ {
		names = append(names, fmt.Sprintf("ma.L%d", i))
	}
	for i := 1; i <= bigQ; i++ {
		names = append(names, fmt.Sprintf("ma.S.L%d", i*period))
	}
	return names
}

// armaNLLObjective evaluates the concentrated NLL as a function of the packed
// factor vector (φ, Φₛ, θ, Θₛ) on the fixed centered series z — the objective
// refineSeasonalMLE minimizes, without the stability penalty (the fitted point
// is strictly inside the stable region; kalmanConcentratedNLL still guards).
func armaNLLObjective(z []float64, p, bigP, q, period int) func([]float64) float64 {
	return func(x []float64) float64 {
		phi, sphi := x[:p], x[p:p+bigP]
		theta, stheta := x[p+bigP:p+bigP+q], x[p+bigP+q:]
		return kalmanConcentratedNLL(z, expandSeasonalAR(phi, sphi, period), expandSeasonalMA(theta, stheta, period))
	}
}

// exogNLLObjective evaluates the concentrated NLL as a function of the packed
// vector (β, φ, Φₛ, θ, Θₛ), re-forming η = y − Xβ and re-centering each call —
// the objective refineExog minimizes.
func exogNLLObjective(series []float64, X [][]float64, k, p, bigP, q, d, bigD, period int) func([]float64) float64 {
	return func(x []float64) float64 {
		beta := x[:k]
		phi, sphi := x[k:k+p], x[k+p:k+p+bigP]
		theta, stheta := x[k+p+bigP:k+p+bigP+q], x[k+p+bigP+q:]
		z := centeredDiff(regressionResiduals(series, X, beta), d, bigD, period)
		return kalmanConcentratedNLL(z, expandSeasonalAR(phi, sphi, period), expandSeasonalMA(theta, stheta, period))
	}
}

// paramCovariance returns cov = 2·H⁻¹ for the fitted coefficient vector coef,
// where H is the numeric Hessian of objective at coef. The factor of 2 converts
// the objective (−2·loglik + const) into the observed information. It returns
// nil when H is singular (standard errors unavailable). H is cloned per solved
// column because gaussian.Solve works in place.
func paramCovariance(objective func([]float64) float64, coef []float64) [][]float64 {
	n := len(coef)
	if n == 0 {
		return [][]float64{}
	}
	H := numHessian(objective, coef)
	cov := make([][]float64, n)
	for i := range cov {
		cov[i] = make([]float64, n)
	}
	for j := 0; j < n; j++ {
		e := make([]float64, n)
		e[j] = 1
		col, err := gaussian.Solve(cloneMatrix(H), e)
		if err != nil {
			return nil
		}
		for i := 0; i < n; i++ {
			cov[i][j] = 2 * col[i]
		}
	}
	return cov
}

func cloneMatrix(m [][]float64) [][]float64 {
	out := make([][]float64, len(m))
	for i := range m {
		out[i] = append([]float64(nil), m[i]...)
	}
	return out
}

// StdErrors returns the standard errors of the fitted coefficients
// (β, φ, Φₛ, θ, Θₛ) in canonical order. It requires an MLE fit
// (WithMethod(MLE)) and returns an error otherwise, or if the information
// matrix was singular. An entry is NaN when its estimated variance is
// non-positive (a weakly identified coefficient).
func (m *ARIMA) StdErrors() ([]float64, error) {
	if !m.mleFit {
		return nil, errors.New("standard errors require an MLE fit (WithMethod(MLE))")
	}
	if m.paramCov == nil {
		return nil, errors.New("standard errors unavailable: singular information matrix")
	}
	se := make([]float64, len(m.paramCov))
	for i := range m.paramCov {
		se[i] = math.Sqrt(m.paramCov[i][i])
	}
	return se, nil
}
