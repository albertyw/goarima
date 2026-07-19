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

func TestSummaryStructure(t *testing.T) {
	series := arSeries(0.6, 400, 2)
	m, _ := NewARIMA(Order{P: 1})
	if err := m.Fit(series, WithMethod(MLE)); err != nil {
		t.Fatal(err)
	}
	s, err := m.Summary()
	if err != nil {
		t.Fatal(err)
	}
	if s.NObs != 400 {
		t.Errorf("NObs=%d want 400", s.NObs)
	}
	// Rows: one per coefficient (here just ar.L1) + a final sigma2 row.
	if len(s.Params) != 2 || s.Params[0].Name != "ar.L1" || s.Params[1].Name != "sigma2" {
		t.Fatalf("rows=%v", s.Params)
	}
	// AIC/BIC match the hand computation with k = p+q+P+Q+exog+1 = 2.
	k := 2.0
	wantAIC := -2*s.LogLik + 2*k
	wantBIC := -2*s.LogLik + k*math.Log(float64(s.NObs))
	if math.Abs(s.AIC-wantAIC) > 1e-9 || math.Abs(s.BIC-wantBIC) > 1e-9 {
		t.Errorf("AIC=%.6f want %.6f; BIC=%.6f want %.6f", s.AIC, wantAIC, s.BIC, wantBIC)
	}
	// z and CI internal consistency for the AR row.
	p := s.Params[0]
	if math.Abs(p.ZScore-p.Coef/p.StdErr) > 1e-9 {
		t.Errorf("z=%.6f want %.6f", p.ZScore, p.Coef/p.StdErr)
	}
	if !(p.CILower < p.Coef && p.Coef < p.CIUpper) {
		t.Errorf("CI [%.4f,%.4f] does not bracket coef %.4f", p.CILower, p.CIUpper, p.Coef)
	}
	if s.String() == "" {
		t.Error("String() empty")
	}
}

func TestSummaryRequiresMLE(t *testing.T) {
	series := arSeries(0.6, 400, 3)
	m, _ := NewARIMA(Order{P: 1})
	_ = m.Fit(series) // HR default
	if _, err := m.Summary(); err == nil {
		t.Error("Summary should error without MLE")
	}
}

func TestStdErrorsExog(t *testing.T) {
	// y_t = 3·x_t + AR(1) noise; deterministic x and seeded noise.
	n := 400
	y := make([]float64, n)
	X := make([][]float64, n)
	r := rand.New(rand.NewSource(4))
	var e float64
	for i := 0; i < n; i++ {
		x := math.Sin(float64(i) / 5)
		e = 0.5*e + r.NormFloat64()
		y[i] = 3*x + e
		X[i] = []float64{x}
	}
	m, _ := NewARIMA(Order{P: 1})
	if err := m.Fit(y, WithExog(X), WithMethod(MLE)); err != nil {
		t.Fatal(err)
	}
	s, err := m.Summary()
	if err != nil {
		t.Fatal(err)
	}
	if s.Params[0].Name != "x1" {
		t.Errorf("first row name=%q want x1", s.Params[0].Name)
	}
	se, _ := m.StdErrors()
	if len(se) != 2 { // β, φ
		t.Fatalf("len(se)=%d want 2", len(se))
	}
	for i, v := range se {
		if !(v > 0) || math.IsInf(v, 0) {
			t.Errorf("se[%d]=%v not finite-positive", i, v)
		}
	}
}
