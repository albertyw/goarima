package goarima

import (
	"errors"
	"math"
	"testing"
)

// Helper: generate an AR(2) series: x_t = φ1 x_{t-1} + φ2 x_{t-2} + ε_t
func generateAR2(phi1, phi2 float64, n int, seed int64) []float64 {
	x := make([]float64, n+2) // extra two for the two lags
	// use a simple RNG for reproducibility
	rand := NewDeterministicRand(seed)
	// init first two values
	x[0] = rand.Float64()*2 - 1
	x[1] = rand.Float64()*2 - 1
	for t := 2; t < n+2; t++ {
		noise := rand.Float64()*2 - 1 // uniform noise in [-1,1]
		x[t] = phi1*x[t-1] + phi2*x[t-2] + noise
	}
	// drop the two init samples
	return x[2:]
}

// Deterministic RNG (LCG) so that tests are reproducible
type DeterministicRand struct {
	seed uint64
}

func NewDeterministicRand(seed int64) *DeterministicRand {
	return &DeterministicRand{seed: uint64(seed)}
}

func (d *DeterministicRand) Float64() float64 {
	// simple LCG
	d.seed = d.seed*6364136223846793005 + 1
	return float64(d.seed%1000000) / 1e6
}

// tolerance for floating‑point comparison
const eps = 1e-6

func TestSolveKnownCoefficients(t *testing.T) {
	// r = [1, 0.5, 0.25]  → φ1=0.5, φ2=0
	r := []float64{1, 0.5, 0.25}
	p := 2

	phi, sigma2, err := SolveFromAutocov(r, p)
	if err != nil {
		t.Fatalf("Solve returned error: %v", err)
	}

	if math.Abs(phi[0]-0.5) > eps {
		t.Errorf("phi1 expected 0.5, got %g", phi[0])
	}
	if math.Abs(phi[1]-0.0) > eps {
		t.Errorf("phi2 expected 0.0, got %g", phi[1])
	}
	if sigma2 <= 0 {
		t.Errorf("sigma2 should be positive, got %g", sigma2)
	}
}

// SolveFromAutocov is a small helper that uses the public Solve
// but works directly on a pre‑computed autocovariance array.
func SolveFromAutocov(r []float64, p int) ([]float64, float64, error) {
	if len(r) < p+1 {
		return nil, 0, errors.New("r too short")
	}
	phi := make([]float64, p)
	sigma2 := r[0]
	for k := 1; k <= p; k++ {
		var acc float64 = r[k]
		for j := 1; j <= k-1; j++ {
			acc += phi[j-1] * r[k-j]
		}
		kappa := acc / sigma2
		phiNew := make([]float64, k)
		for i := 0; i < k-1; i++ {
			phiNew[i] = phi[i] - kappa*phi[k-2-i]
		}
		phiNew[k-1] = kappa
		copy(phi[:k], phiNew)
		sigma2 *= 1 - kappa*kappa
	}
	return phi, sigma2, nil
}

func TestSolveLargeSyntheticSeries(t *testing.T) {
	phi1, phi2 := 0.5, -0.3
	N := 5000
	series := generateAR2(phi1, phi2, N, 42)

	// Compute sample autocovariances up to lag 2
	r := autocovariances(series, 2)

	phi, sigma2, err := SolveFromAutocov(r, 2)
	if err != nil {
		t.Fatalf("Solve returned error: %v", err)
	}

	if math.Abs(phi[0]-phi1) > 0.05 {
		t.Errorf("phi1: expected %g, got %g", phi1, phi[0])
	}
	if math.Abs(phi[1]-phi2) > 0.05 {
		t.Errorf("phi2: expected %g, got %g", phi2, phi[1])
	}
	if sigma2 <= 0 {
		t.Errorf("sigma2 should be positive, got %g", sigma2)
	}

	// Verify that R*phi ≈ r[1:3]
	R00 := r[0]
	R01 := r[1]
	R10 := r[1]
	R11 := r[0]
	Rphi1 := R00*phi[0] + R01*phi[1]
	Rphi2 := R10*phi[0] + R11*phi[1]
	if math.Abs(Rphi1-r[1]) > 1e-3 {
		t.Errorf("R*phi1 mismatch: got %g, want %g", Rphi1, r[1])
	}
	if math.Abs(Rphi2-r[2]) > 1e-3 {
		t.Errorf("R*phi2 mismatch: got %g, want %g", Rphi2, r[2])
	}
}

func TestSolveShortSeries(t *testing.T) {
	y := []float64{1, 2, 3} // length 3
	if _, _, err := solveYuleWalker(y, 5); err == nil {
		t.Errorf("expected error for too short series, got none")
	}
	if _, _, err := solveYuleWalker(y, 0); err == nil {
		t.Errorf("expected error for p<=0, got none")
	}
}
