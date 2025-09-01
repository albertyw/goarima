package goarima

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/albertyw/gaussian"
)

var output io.Writer

func init() {
	output = os.Stdout
}

// autocorrelationAtLag calculates the autocorrelation at a given lag for a given time series.
// This function has been manually verified.
//
// Args:
//
//	series: The time series data.
//	lag: The lag at which to calculate the autocorrelation.
//
// Returns:
//
//	The autocorrelation at the specified lag. Returns 0 if the lag is greater than or equal to the length of the series.
func autocorrelationAtLag(series []float64, lag int) float64 {
	n := len(series)
	if lag >= n {
		return 0
	}

	// Calculate mean
	var sum float64
	for _, v := range series {
		sum += v
	}
	mean := sum / float64(n)

	// Calculate numerator (covariance)
	var numerator float64
	for i := 0; i < n-lag; i++ {
		numerator += (series[i] - mean) * (series[i+lag] - mean)
	}

	// Calculate denominator (variance)
	var denominator float64
	for _, x := range series {
		denominator += (x - mean) * (x - mean)
	}

	// Calculate autocorrelation
	return numerator / float64(n)
}

// buildAutocorrelationVector builds a vector of autocorrelations for a given time series up to a specified order.
//
// Args:
//
//	series: The time series data.
//	order: The maximum order of the autocorrelation to calculate.
//
// Returns:
//
//	A slice of floats representing the autocorrelations from lag 0 to lag 'order'.
func buildAutocorrelationVector(series []float64, order int) []float64 {
	rVec := make([]float64, order+1)
	for k := 0; k <= order; k++ {
		rVec[k] = autocorrelationAtLag(series, k)
	}
	return rVec
}

// buildToeplitzMatrix builds a Toeplitz matrix from a given autocorrelation vector.
// This function has been manually verified against scipy.linalg.toeplitz.
//
// Args:
//
//	rVec: The autocorrelation vector.
//
// Returns:
//
//	A 2D slice of floats representing the (n-1) x (n-1) Toeplitz matrix.
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

// buildRHSVector builds the right-hand side vector for the Yule-Walker equations.
//
// Args:
//
//	rVec: The autocorrelation vector.
//
// Returns:
//
//	A slice of floats representing the right-hand side vector.
func buildRHSVector(rVec []float64) []float64 {
	bVec := make([]float64, len(rVec)-1)
	copy(bVec, rVec[1:])
	return bVec
}

// solveYuleWalker applies the Yule‑Walker equations to the input series `y`
// and returns the AR(p) coefficients and the white‑noise variance.
// The series is assumed to be already differenced (i.e. stationary).
//
//	R φ = r
//
//	R   = Toeplitz(r0, r1, …, r(p‑1))
//	r   = [r1, r2, …, rp]^T
//
//	σ² = r0 – φᵀ r   (residual variance)
//
// Assumptions:
//   - The input series 'series' is stationary.  This means its statistical properties
//     (mean, variance, autocorrelation) do not change over time.  It is often
//     necessary to difference the series before applying the Yule-Walker equations
//     to achieve stationarity.
//
// Args:
//
//	series: The stationary time series data.
//	order: The order (p) of the autoregressive model.
//
// Returns:
//
//	coeffs: The AR(p) coefficients.
//	sigma2: The estimated white-noise variance (residual variance).
//	error: An error if the input is invalid or the linear system cannot be solved.
func solveYuleWalker(series []float64, order int) ([]float64, float64, error) {
	if order <= 0 || order >= len(series) {
		return nil, 0, errors.New("solveYuleWalker: order must be >0 and <len(series)")
	}

	// a. Autocorrelation vector r[0 … order]
	rVec := buildAutocorrelationVector(series, order)

	// b. Toeplitz matrix R
	R := buildToeplitzMatrix(rVec)

	// c. RHS vector b
	b := buildRHSVector(rVec)

	fmt.Fprintf(output, "R:\n")
	for i := range R {
		fmt.Fprintf(output, "%v\n", R[i])
	}
	fmt.Fprintf(output, "b:\n")
	fmt.Fprintf(output, "%v\n", b)

	// d. Solve R * a = b
	coeffs, err := gaussian.Solve(R, b)
	if err != nil {
		return nil, 0, err
	}

	// e. Estimate σ² = r0 – Σ a_i * r_i
	var sigma2 float64
	for i, a := range coeffs {
		sigma2 += a * rVec[i+1] // r[1]…r[order]
	}
	sigma2 = rVec[0] - sigma2

	return coeffs, sigma2, nil
}
