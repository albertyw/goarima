package goarima

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// rampWithNoise returns a linear trend plus white noise (needs one difference).
func rampWithNoise(n int, slope float64, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	s := make([]float64, n)
	for i := range s {
		s[i] = slope*float64(i) + r.NormFloat64()
	}
	return s
}

// randomWalk returns a cumulative sum of white noise (needs one difference).
func randomWalk(n int, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	s := make([]float64, n)
	for i := 1; i < n; i++ {
		s[i] = s[i-1] + r.NormFloat64()
	}
	return s
}

func TestSelectD(t *testing.T) {
	testCases := []struct {
		name   string
		series []float64
		maxD   int
		want   int
	}{
		{"white noise stays at 0", genAR1(500, 0, 1), 2, 0},
		{"linear ramp needs 1", rampWithNoise(500, 0.5, 2), 2, 1},
		{"random walk needs 1", randomWalk(500, 3), 2, 1},
		{"constant stays at 0", make([]float64, 50), 2, 0},
		{"maxD 0 forces 0", rampWithNoise(500, 0.5, 4), 0, 0},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, selectD(tc.series, tc.maxD))
		})
	}
}

func TestAutoARIMAWhiteNoise(t *testing.T) {
	series := genAR1(400, 0, 5)
	model, err := AutoARIMA(series, 3, 2, 3)
	require.NoError(t, err)
	require.NotNil(t, model)

	p, d, q := model.Orders()
	assert.Equal(t, 0, d) // white noise should not be differenced
	assert.LessOrEqual(t, p, 3)
	assert.LessOrEqual(t, q, 3)

	forecast, err := model.Forecast(5)
	require.NoError(t, err)
	assert.Len(t, forecast, 5)
	for _, f := range forecast {
		assert.False(t, math.IsNaN(f) || math.IsInf(f, 0))
	}
}

func TestAutoARIMATrendChoosesDifferencing(t *testing.T) {
	series := rampWithNoise(400, 0.7, 6)
	model, err := AutoARIMA(series, 3, 2, 3)
	require.NoError(t, err)
	_, d, _ := model.Orders()
	assert.GreaterOrEqual(t, d, 1)

	// A trend forecast should keep climbing, not flatten out.
	forecast, err := model.Forecast(3)
	require.NoError(t, err)
	assert.Greater(t, forecast[2], series[len(series)-1])
}

func TestAutoARIMARespectsBounds(t *testing.T) {
	series := genARMA11(400, 0.5, 0.3, 7)
	model, err := AutoARIMA(series, 2, 1, 2)
	require.NoError(t, err)
	p, d, q := model.Orders()
	assert.LessOrEqual(t, p, 2)
	assert.LessOrEqual(t, d, 1)
	assert.LessOrEqual(t, q, 2)
	assert.True(t, p > 0 || q > 0) // (0,0) is never selected
}

func TestAutoARIMAErrors(t *testing.T) {
	_, err := AutoARIMA([]float64{1.0}, 1, 1, 1)
	assert.Error(t, err)

	_, err = AutoARIMA(genAR1(100, 0.3, 8), -1, 1, 1)
	assert.Error(t, err)
}

func TestAutoARIMARejectsNonFiniteInput(t *testing.T) {
	// A NaN in the input makes every candidate fit degenerate (the series is
	// misclassified as constant, sigma2 floors, and the grid "selects" a model
	// that forecasts NaN). AutoARIMA must reject non-finite input up front.
	series := genAR1(100, 0.3, 9)
	series[50] = math.NaN()
	_, err := AutoARIMA(series, 2, 1, 2)
	assert.ErrorContains(t, err, "non-finite")
}

func TestAutoARIMAWithCSSRefinement(t *testing.T) {
	// AutoARIMA must accept and thread the refinement option through to its
	// fits, still returning a usable, finite-forecasting model.
	series := genARMA11(500, 0.5, 0.4, 2)
	model, err := AutoARIMA(series, 3, 1, 3, WithCSSRefinement())
	require.NoError(t, err)
	require.NotNil(t, model)

	forecast, err := model.Forecast(5)
	require.NoError(t, err)
	for _, f := range forecast {
		assert.False(t, math.IsNaN(f) || math.IsInf(f, 0))
	}
}

func TestAutoARIMAWithMLE(t *testing.T) {
	// AutoARIMA must accept and thread the MLE option through to its fits, still
	// returning a usable, finite-forecasting model.
	series := genARMA11(500, 0.5, 0.4, 2)
	model, err := AutoARIMA(series, 3, 1, 3, WithMLE())
	require.NoError(t, err)
	require.NotNil(t, model)

	forecast, err := model.Forecast(5)
	require.NoError(t, err)
	for _, f := range forecast {
		assert.False(t, math.IsNaN(f) || math.IsInf(f, 0))
	}
}

func TestAIC(t *testing.T) {
	// Lower residual variance gives a lower (better) AIC.
	assert.Less(t, aic(100, 0.5, 1, 0), aic(100, 1.0, 1, 0))
	// For equal variance, more parameters gives a higher AIC.
	assert.Less(t, aic(100, 1.0, 1, 0), aic(100, 1.0, 2, 1))
	// Degenerate (zero) variance stays finite.
	assert.False(t, math.IsInf(aic(100, 0, 1, 1), 0))
}
