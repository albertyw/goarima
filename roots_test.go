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
