package goarima

import (
	"math"
	"testing"
)

func TestNumHessianQuadratic(t *testing.T) {
	// f(x) = 0.5 * xᵀ A x, A symmetric ⇒ Hessian = A everywhere.
	A := [][]float64{{2, 0.5}, {0.5, 3}}
	f := func(x []float64) float64 {
		var s float64
		for i := range x {
			for j := range x {
				s += 0.5 * x[i] * A[i][j] * x[j]
			}
		}
		return s
	}
	H := numHessian(f, []float64{0.4, -0.2})
	for i := range A {
		for j := range A {
			if math.Abs(H[i][j]-A[i][j]) > 1e-4 {
				t.Errorf("H[%d][%d]=%.6f want %.6f", i, j, H[i][j], A[i][j])
			}
		}
	}
}
