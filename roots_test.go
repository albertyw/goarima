package goarima

import (
	"math"
	"testing"
)

func TestRepairStationaryAR1(t *testing.T) {
	// AR(1) phi=1.5: root at 1/1.5≈0.667 (inside) -> reflect to 1.5 -> phi'≈0.667.
	got := repairStationary([]float64{1.5})
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
	if math.Abs(got[0]-1.0/1.5) > 1e-6 {
		t.Errorf("phi' = %v, want ≈%v", got[0], 1.0/1.5)
	}
	if !isStationary(got) {
		t.Errorf("repaired AR not stationary: %v", got)
	}
}

func TestRepairInvertibleMA1(t *testing.T) {
	got := repairInvertible([]float64{1.5})
	if math.Abs(got[0]-1.0/1.5) > 1e-6 {
		t.Errorf("theta' = %v, want ≈%v", got[0], 1.0/1.5)
	}
	if !isInvertible(got) {
		t.Errorf("repaired MA not invertible: %v", got)
	}
}

func TestRepairStationaryComplexRoots(t *testing.T) {
	// AR(2) with complex non-stationary roots; repaired must be stationary and real.
	in := []float64{0.2, 0.95}
	got := repairStationary(in)
	if isStationary(in) {
		t.Fatal("test input is already stationary; pick an unstable one")
	}
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2", len(got))
	}
	for _, v := range got {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite coefficient: %v", got)
		}
	}
	if !isStationary(got) {
		t.Errorf("repaired AR(2) not stationary: %v", got)
	}
}

func TestRepairIdempotentOnStable(t *testing.T) {
	in := []float64{0.5}
	got := repairStationary(in)
	if math.Abs(got[0]-0.5) > 1e-12 {
		t.Errorf("stable input changed: %v", got)
	}
	inMA := []float64{0.4}
	gotMA := repairInvertible(inMA)
	if math.Abs(gotMA[0]-0.4) > 1e-12 {
		t.Errorf("stable MA input changed: %v", gotMA)
	}
}

func TestRepairEmpty(t *testing.T) {
	if got := repairStationary(nil); len(got) != 0 {
		t.Errorf("nil AR -> %v, want empty", got)
	}
	if got := repairInvertible([]float64{}); len(got) != 0 {
		t.Errorf("empty MA -> %v, want empty", got)
	}
}

func TestPolyRootsConstant(t *testing.T) {
	// Constant / degree-0 polynomials have no roots.
	if got := polyRoots([]float64{1}); got != nil {
		t.Errorf("polyRoots([1]) = %v, want nil", got)
	}
	if got := polyRoots(nil); got != nil {
		t.Errorf("polyRoots(nil) = %v, want nil", got)
	}
}

// unstableSeries is a deterministic series whose Hannan-Rissanen Stage-2 OLS
// lands outside the valid region: (0,0,1) is non-invertible and (2,0,2) is
// non-stationary, so it exercises both repair branches.
func unstableSeries() []float64 {
	s := make([]float64, 120)
	for i := range s {
		s[i] = math.Sin(float64(i)*1.7) + 0.9*math.Sin(float64(i)*0.85)
	}
	return s
}

func TestFitRejectsUnstableByDefault(t *testing.T) {
	s := unstableSeries()
	m, err := NewARIMA(0, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Fit(s); err == nil {
		t.Fatal("expected default Fit to reject the non-invertible MA fit")
	}
}

func TestFitRepairsNonInvertibleMA(t *testing.T) {
	s := unstableSeries()
	m, err := NewARIMA(0, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Fit(s, WithRootRepair()); err != nil {
		t.Fatalf("WithRootRepair Fit failed: %v", err)
	}
	if !isInvertible(m.Theta()) {
		t.Errorf("theta not invertible after repair: %v", m.Theta())
	}
	fc, err := m.Forecast(5)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range fc {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite forecast: %v", fc)
		}
	}
}

func TestFitRepairsNonStationaryAR(t *testing.T) {
	s := unstableSeries()
	m, err := NewARIMA(2, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Fit(s, WithRootRepair()); err != nil {
		t.Fatalf("WithRootRepair Fit failed: %v", err)
	}
	if !isStationary(m.Phi()) {
		t.Errorf("phi not stationary after repair: %v", m.Phi())
	}
}

func TestFitRepairsSeasonal(t *testing.T) {
	s := unstableSeries()
	m, err := NewSARIMA(0, 0, 1, 0, 0, 1, 12)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Fit(s, WithRootRepair()); err != nil {
		t.Fatalf("seasonal WithRootRepair Fit failed: %v", err)
	}
	if !isInvertible(m.Theta()) || !isInvertible(m.SeasonalTheta()) {
		t.Errorf("seasonal MA not invertible after repair: %v %v", m.Theta(), m.SeasonalTheta())
	}
}
