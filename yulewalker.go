package goarima

import (
	"errors"
	"math"

	"github.com/albertyw/gaussian"
)

// autocovarianceAtLag returns the biased sample autocovariance of the series
// at the given lag,
//
//	γ̂_k = (1/n)·Σ_{t=1}^{n-k} (y_t − ȳ)(y_{t+k} − ȳ)
//
// or 0 when the lag is at or beyond the series length. The biased (divide by
// n, not n-k) estimator keeps the Toeplitz system positive definite, so the
// Yule-Walker AR estimate is always stationary.
func autocovarianceAtLag(series []float64, lag int) float64 {
	n := len(series)
	if lag >= n {
		return 0
	}

	m := mean(series)
	var sum float64
	for i := 0; i < n-lag; i++ {
		sum += (series[i] - m) * (series[i+lag] - m)
	}
	return sum / float64(n)
}

// buildAutocovarianceVector returns the autocovariances [γ̂_0, γ̂_1, …, γ̂_order]
// of the series.
func buildAutocovarianceVector(series []float64, order int) []float64 {
	rVec := make([]float64, order+1)
	for k := 0; k <= order; k++ {
		rVec[k] = autocovarianceAtLag(series, k)
	}
	return rVec
}

// buildToeplitzMatrix builds the p×p Toeplitz matrix R of the Yule-Walker
// system from the autocovariance vector rVec = [γ_0 … γ_p]: R[i][j] = γ_|i−j|.
func buildToeplitzMatrix(rVec []float64) [][]float64 {
	p := len(rVec) - 1 // number of AR coefficients
	matrix := make([][]float64, p)
	for i := 0; i < p; i++ {
		matrix[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			index := int(math.Abs(float64(i - j)))
			matrix[i][j] = rVec[index]
		}
	}
	return matrix
}

// buildRHSVector builds the right-hand side [γ_1 … γ_p] of the Yule-Walker
// equations from the autocovariance vector rVec = [γ_0 … γ_p].
func buildRHSVector(rVec []float64) []float64 {
	bVec := make([]float64, len(rVec)-1)
	copy(bVec, rVec[1:])
	return bVec
}

// solveYuleWalker fits an AR(order) model to the series by solving the
// Yule-Walker equations
//
//	R φ = r
//
//	R   = Toeplitz(r0, r1, …, r(p‑1))
//	r   = [r1, r2, …, rp]^T
//
//	σ² = r0 – φᵀ r   (residual variance)
//
// and returns the AR coefficients and the white-noise variance σ². The series
// must already be stationary (difference it first if necessary).
func solveYuleWalker(series []float64, order int) ([]float64, float64, error) {
	if order <= 0 || order >= len(series) {
		return nil, 0, errors.New("solveYuleWalker: order must be >0 and <len(series)")
	}

	// Autocovariance vector r[0 … order], then solve from it.
	rVec := buildAutocovarianceVector(series, order)
	return solveYuleWalkerFromAutocov(rVec, order)
}

// solveYuleWalkerFromAutocov solves the Yule‑Walker equations directly from an
// autocovariance vector rVec = [r0, r1, …, r_order] and returns the AR(order)
// coefficients and the white‑noise variance σ² = r0 – Σ aᵢ·rᵢ.
//
// A constant series has r0 == 0, which makes the Toeplitz system singular; in
// that case the AR coefficients are taken to be zero with zero variance.
func solveYuleWalkerFromAutocov(rVec []float64, order int) ([]float64, float64, error) {
	if order <= 0 || order >= len(rVec) {
		return nil, 0, errors.New("solveYuleWalkerFromAutocov: order must be >0 and <len(rVec)")
	}

	// Degenerate (constant) series: no autocovariance to solve against.
	if math.Abs(rVec[0]) < 1e-12 {
		return make([]float64, order), 0, nil
	}

	// Toeplitz matrix R and RHS vector b, then solve R * a = b.
	R := buildToeplitzMatrix(rVec)
	b := buildRHSVector(rVec)
	coeffs, err := gaussian.Solve(R, b)
	if err != nil {
		return nil, 0, err
	}

	// Estimate σ² = r0 – Σ aᵢ · rᵢ.
	var sigma2 float64
	for i, a := range coeffs {
		sigma2 += a * rVec[i+1] // r[1]…r[order]
	}
	sigma2 = rVec[0] - sigma2

	return coeffs, sigma2, nil
}
