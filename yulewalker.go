package goarima

import (
	"errors"
	"fmt"
)

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
func solveYuleWalker(y []float64, p int) ([]float64, float64, error) {
	if p <= 0 {
		return nil, 0, errors.New("order p must be > 0")
	}
	if len(y) < p+1 {
		return nil, 0, errors.New("series too short for the requested order")
	}

	// 1. Compute autocovariances r[0..p]
	r := autocovariances(y, p)

	// 2. Levinson–Durbin recursion
	phi := make([]float64, p) // φ1…φp
	sigma2 := r[0]            // σ² at order 0

	for k := 1; k <= p; k++ {
		// Compute the reflection coefficient κ
		var acc float64 = r[k]
		for j := 1; j <= k-1; j++ {
			acc += phi[j-1] * r[k-j]
		}
		kappa := acc / sigma2

		// Update the AR coefficients
		phiNew := make([]float64, k)
		for i := 0; i < k-1; i++ {
			phiNew[i] = phi[i] - kappa*phi[k-2-i]
		}
		phiNew[k-1] = kappa

		copy(phi[:k], phiNew)
		sigma2 *= 1 - kappa*kappa
	}

	fmt.Println("Yule-Walker coefficients:", phi)
	fmt.Println("White noise variance:", sigma2)
	return phi, sigma2, nil
}

// autocovariances returns r[0..p] where
//
//	r[k] = (1/N) Σ_{t=k+1}^{N} y_t * y_{t-k}
func autocovariances(y []float64, p int) []float64 {
	N := len(y)
	r := make([]float64, p+1)
	for k := 0; k <= p; k++ {
		var sum float64
		for t := k; t < N; t++ {
			sum += y[t] * y[t-k]
		}
		r[k] = sum / float64(N)
	}
	return r
}
