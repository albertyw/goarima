package goarima

import (
	"math"
	"math/rand"
	"testing"
)

// arSeries returns a deterministic AR(1) series of length n with coefficient
// phi, driven by a seeded Gaussian (no network/file I/O).
func arSeries(phi float64, n int, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float64, n)
	for i := 1; i < n; i++ {
		out[i] = phi*out[i-1] + r.NormFloat64()
	}
	return out
}

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

func TestStdErrorsRequireMLE(t *testing.T) {
	series := arSeries(0.6, 400, 1)
	for _, tc := range []struct {
		name string
		opts []FitOption
	}{
		{"default-HR", nil},
		{"CSS", []FitOption{WithMethod(CSS)}},
	} {
		m, _ := NewARIMA(Order{P: 1})
		if err := m.Fit(series, tc.opts...); err != nil {
			t.Fatalf("%s fit: %v", tc.name, err)
		}
		if _, err := m.StdErrors(); err == nil {
			t.Errorf("%s: StdErrors should error without MLE", tc.name)
		}
	}
	// Unfit model errors too.
	m, _ := NewARIMA(Order{P: 1})
	if _, err := m.StdErrors(); err == nil {
		t.Error("unfit StdErrors should error")
	}
}

func TestStdErrorsAR1CloseToClosedForm(t *testing.T) {
	const phi, n = 0.6, 500
	series := arSeries(phi, n, 1)
	m, _ := NewARIMA(Order{P: 1})
	if err := m.Fit(series, WithMethod(MLE)); err != nil {
		t.Fatal(err)
	}
	se, err := m.StdErrors()
	if err != nil {
		t.Fatal(err)
	}
	// Asymptotic SE(φ̂) = sqrt((1−φ²)/n).
	want := math.Sqrt((1 - phi*phi) / n)
	if len(se) != 1 || se[0] <= 0 || math.Abs(se[0]-want)/want > 0.30 {
		t.Errorf("SE(phi)=%v want ≈ %.5f (±30%%)", se, want)
	}
}
