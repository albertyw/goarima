package goarima

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// expandSeasonalAR/MA expand the multiplicative seasonal factor into the
// recursion-coefficient array the forecast/state-space code consumes.

func TestExpandSeasonalARMultipliesFactors(t *testing.T) {
	// (1 − 0.5B)(1 − 0.3B⁴) = 1 − 0.5B − 0.3B⁴ + 0.15B⁵.
	// Recursion coeffs a (y_t = Σ a_i y_{t-i}): a₁=0.5, a₄=0.3, a₅=−0.15.
	got := expandSeasonalAR([]float64{0.5}, []float64{0.3}, 4)
	assert.InDeltaSlice(t, []float64{0.5, 0, 0, 0.3, -0.15}, got, 1e-12)
}

func TestExpandSeasonalMAMultipliesFactors(t *testing.T) {
	// (1 + 0.4B)(1 + 0.2B⁴) = 1 + 0.4B + 0.2B⁴ + 0.08B⁵.
	got := expandSeasonalMA([]float64{0.4}, []float64{0.2}, 4)
	assert.InDeltaSlice(t, []float64{0.4, 0, 0, 0.2, 0.08}, got, 1e-12)
}

func TestExpandSeasonalARNoSeasonalReturnsRegular(t *testing.T) {
	got := expandSeasonalAR([]float64{0.5, -0.2}, nil, 12)
	assert.InDeltaSlice(t, []float64{0.5, -0.2}, got, 1e-12)
}

func TestExpandSeasonalMANoRegularIsSeasonalOnly(t *testing.T) {
	// Pure seasonal MA(1): (1 + 0.3B¹²) → coeff 0.3 at lag 12, zeros before.
	got := expandSeasonalMA(nil, []float64{0.3}, 12)
	want := make([]float64, 12)
	want[11] = 0.3
	assert.InDeltaSlice(t, want, got, 1e-12)
}

func TestExpandSeasonalEmptyIsEmpty(t *testing.T) {
	assert.Empty(t, expandSeasonalAR(nil, nil, 12))
}

func TestNewSARIMAStoresSeasonalOrders(t *testing.T) {
	m, err := NewSARIMA(1, 0, 1, 2, 1, 1, 12) // p,d,q,P,D,Q,m
	assert.NoError(t, err)
	P, D, Q, period := m.SeasonalOrders()
	assert.Equal(t, 2, P)
	assert.Equal(t, 1, D)
	assert.Equal(t, 1, Q)
	assert.Equal(t, 12, period)
}

func TestNewSARIMASeasonalCoeffGettersLength(t *testing.T) {
	m, err := NewSARIMA(1, 0, 1, 2, 0, 1, 12)
	assert.NoError(t, err)
	assert.Len(t, m.SeasonalPhi(), 2)
	assert.Len(t, m.SeasonalTheta(), 1)
}

func TestNewSARIMARejectsSeasonalARWithoutValidPeriod(t *testing.T) {
	_, err := NewSARIMA(0, 0, 0, 1, 0, 0, 1) // P=1 but m<2
	assert.Error(t, err)
}

func TestNewSARIMARejectsNegativeSeasonalOrders(t *testing.T) {
	_, err := NewSARIMA(1, 0, 0, -1, 0, 0, 12)
	assert.Error(t, err)
	_, err = NewSARIMA(1, 0, 0, 0, 0, -1, 12)
	assert.Error(t, err)
}
