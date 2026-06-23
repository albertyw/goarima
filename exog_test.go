package goarima

import (
	"math"
	"math/rand"
	"testing"
)

func TestValidateExogMatrix(t *testing.T) {
	good := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	k, err := validateExogMatrix(good, 3)
	if err != nil || k != 2 {
		t.Fatalf("good matrix: k=%d err=%v", k, err)
	}
	if _, err := validateExogMatrix(good, 4); err == nil {
		t.Error("row-count mismatch should error")
	}
	ragged := [][]float64{{1, 2}, {3}}
	if _, err := validateExogMatrix(ragged, 2); err == nil {
		t.Error("ragged rows should error")
	}
	nan := [][]float64{{math.NaN()}}
	if _, err := validateExogMatrix(nan, 1); err == nil {
		t.Error("non-finite entry should error")
	}
}

func TestOLSBetaRecoversKnownSlope(t *testing.T) {
	// y = 3*x1 - 2*x2 exactly; OLS must recover (3, -2).
	dX := [][]float64{{1, 0}, {0, 1}, {1, 1}, {2, 1}, {1, 3}}
	dy := make([]float64, len(dX))
	for i, r := range dX {
		dy[i] = 3*r[0] - 2*r[1]
	}
	beta, err := olsBeta(dy, dX)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(beta[0]-3) > 1e-9 || math.Abs(beta[1]+2) > 1e-9 {
		t.Fatalf("got beta=%v, want [3 -2]", beta)
	}
}

func TestDifferenceExogMatchesColumnwise(t *testing.T) {
	X := [][]float64{{1, 10}, {2, 14}, {4, 20}, {7, 28}}
	got := differenceExog(X, 1, 0, 0) // first regular difference
	want := [][]float64{{1, 4}, {2, 6}, {3, 8}}
	if len(got) != len(want) {
		t.Fatalf("rows: got %d want %d", len(got), len(want))
	}
	for i := range want {
		for j := range want[i] {
			if math.Abs(got[i][j]-want[i][j]) > 1e-12 {
				t.Fatalf("row %d col %d: got %v want %v", i, j, got[i][j], want[i][j])
			}
		}
	}
}

func TestFitWithExogRecoversBeta(t *testing.T) {
	// y_t = 2*x_t + AR(1) errors with phi=0.5 and white-noise innovations.
	rng := rand.New(rand.NewSource(20260622))
	n := 400
	x := make([]float64, n)
	y := make([]float64, n)
	var eta float64
	for i := 0; i < n; i++ {
		x[i] = math.Sin(float64(i) / 5.0)
		eta = 0.5*eta + rng.NormFloat64()*0.3
		y[i] = 2*x[i] + eta
	}
	X := make([][]float64, n)
	for i := range X {
		X[i] = []float64{x[i]}
	}
	m, err := NewARIMA(1, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Fit(y, WithExog(X)); err != nil {
		t.Fatal(err)
	}
	b := m.Beta()
	if len(b) != 1 || math.Abs(b[0]-2) > 0.2 {
		t.Fatalf("got beta=%v, want ~[2]", b)
	}
	if len(m.Phi()) != 1 || math.Abs(m.Phi()[0]-0.5) > 0.2 {
		t.Fatalf("got phi=%v, want ~[0.5]", m.Phi())
	}
}

func TestFitWithoutExogLeavesBetaEmpty(t *testing.T) {
	m, _ := NewARIMA(1, 0, 0)
	if err := m.Fit([]float64{1, 2, 1, 2, 1, 2, 1, 2, 1, 2}); err != nil {
		t.Fatal(err)
	}
	if len(m.Beta()) != 0 {
		t.Fatalf("Beta should be empty without exog, got %v", m.Beta())
	}
}
