package goarima

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ar1Series returns a stationary AR(1) series x_t = phi·x_{t-1} + e (d=0).
func ar1Series(n int, phi float64, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	s := make([]float64, n)
	for i := 1; i < n; i++ {
		s[i] = phi*s[i-1] + r.NormFloat64()
	}
	return s
}

func TestWithStepwiseSetsConfig(t *testing.T) {
	var cfg fitConfig
	assert.False(t, cfg.stepwise, "default is the exhaustive grid")
	WithStepwise().applyAuto(&cfg)
	assert.True(t, cfg.stepwise)
}

func TestStepwiseFindsGridOptimumOnCleanAR1(t *testing.T) {
	// On a smooth strongly-AR(1) series, the stepwise hill-climb should reach
	// the same global optimum the exhaustive grid finds.
	series := ar1Series(300, 0.7, 11)
	grid, err := AutoARIMA(series, Bounds{MaxP: 4, MaxD: 2, MaxQ: 4})
	require.NoError(t, err)
	step, err := AutoARIMA(series, Bounds{MaxP: 4, MaxD: 2, MaxQ: 4}, WithStepwise())
	require.NoError(t, err)

	assert.Equal(t, grid.Order(), step.Order())
}

func TestStepwiseProducesValidFit(t *testing.T) {
	// Stepwise is a heuristic; on every example-like series it must still return
	// a fitted, invertible model with a finite forecast.
	series := rampWithNoise(200, 0.5, 3)
	model, err := AutoARIMA(series, Bounds{MaxP: 5, MaxD: 2, MaxQ: 5}, WithStepwise())
	require.NoError(t, err)
	o := model.Order()
	assert.LessOrEqual(t, o.P, 5)
	assert.LessOrEqual(t, o.Q, 5)
	assert.False(t, o.P == 0 && o.Q == 0, "(0,0) is never selected")
	fc, err := model.Forecast(6)
	require.NoError(t, err)
	for _, f := range fc {
		assert.False(t, math.IsNaN(f) || math.IsInf(f, 0))
	}
}

func TestStepwiseIsDeterministic(t *testing.T) {
	series := ar1Series(250, 0.5, 99)
	a, err := AutoARIMA(series, Bounds{MaxP: 4, MaxD: 2, MaxQ: 4}, WithStepwise())
	require.NoError(t, err)
	b, err := AutoARIMA(series, Bounds{MaxP: 4, MaxD: 2, MaxQ: 4}, WithStepwise())
	require.NoError(t, err)
	assert.Equal(t, a.Phi(), b.Phi())
	assert.Equal(t, a.Theta(), b.Theta())
}

func TestStepwiseRespectsZeroMaxima(t *testing.T) {
	// With maxP=maxQ=0 the only non-(0,0) order is unreachable, so selection
	// fails exactly as the grid does.
	series := ar1Series(120, 0.6, 5)
	_, err := AutoARIMA(series, Bounds{MaxP: 0, MaxD: 0, MaxQ: 0}, WithStepwise())
	assert.Error(t, err)
}
