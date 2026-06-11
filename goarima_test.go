package goarima

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRandomWalkFits(t *testing.T) {
	// ARIMA(0,1,0) is the random-walk(-with-drift) baseline: no AR or MA terms,
	// just differencing. It must fit and forecast, not error out.
	series := []float64{1, 3, 2, 5, 4, 7, 6, 9, 8, 11}
	model, err := NewARIMA(0, 1, 0)
	require.NoError(t, err)
	require.NoError(t, model.Fit(series))

	forecast, err := model.Forecast(3)
	require.NoError(t, err)
	require.Len(t, forecast, 3)
	// drift = mean of first differences = (11-1)/9; forecast extrapolates it.
	drift := (11.0 - 1.0) / 9.0
	assert.InDelta(t, 11+drift, forecast[0], 1e-9)
	assert.InDelta(t, 11+2*drift, forecast[1], 1e-9)
	assert.InDelta(t, 11+3*drift, forecast[2], 1e-9)
}

func TestNonInvertibleReturnsError(t *testing.T) {
	// First-differencing this series gives a strongly negatively autocorrelated
	// sequence, so an MA(1) fit lands outside the invertible region. The model
	// must reject it with an error rather than return a diverging forecast.
	series := []float64{1, 3, 2, 5, 4, 7, 6, 9, 8, 11}
	model, err := NewARIMA(0, 1, 1)
	require.NoError(t, err)
	assert.Error(t, model.Fit(series))
}

func TestSimpleARIMA(t *testing.T) {
	var testCases = []struct {
		name     string
		data     []float64
		p, d, q  int
		expected []float64
	}{
		{
			// A stationary AR(1) fitted to this series gets phi = -0.9 and mean
			// 1.5, so the forecast is a damped oscillation decaying toward 1.5
			// rather than an exact repetition of the 1,2 pattern.
			name:     "ARIMA(1,0,0) with oscillating data",
			data:     []float64{1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
			p:        1,
			d:        0,
			q:        0,
			expected: []float64{1.05, 1.905, 1.1355, 1.82805, 1.204755},
		},
		{
			name:     "ARIMA(1,1,1) with simple data",
			data:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			p:        1,
			d:        1,
			q:        1,
			expected: []float64{11, 12, 13, 14, 15},
		},
		{
			name:     "ARIMA(1,0,0) with simple data",
			data:     []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			p:        1,
			d:        0,
			q:        0,
			expected: []float64{1, 1, 1, 1, 1},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model, err := NewARIMA(tc.p, tc.d, tc.q)
			require.NoError(t, err)
			require.NotNil(t, model)
			err = model.Fit(tc.data)
			require.NoError(t, err)
			forecast, err := model.Forecast(5)
			require.NoError(t, err)
			assert.Equal(t, 5, len(forecast))
			for i := range forecast {
				assert.InDelta(t, tc.expected[i], forecast[i], 1e-6)
			}
		})
	}
}

func TestNewARIMAErrors(t *testing.T) {
	_, err := NewARIMA(-1, 0, 0)
	assert.Error(t, err)
	_, err = NewARIMA(0, 0, 0)
	assert.Error(t, err)
}

func TestFitRejectsNonFiniteInput(t *testing.T) {
	// NaN compares false against every threshold, so without an explicit guard a
	// NaN-bearing series is misclassified as constant and "fits" successfully,
	// silently forecasting NaN. Fit must reject non-finite input instead.
	testCases := []struct {
		name string
		bad  float64
	}{
		{"NaN", math.NaN()},
		{"+Inf", math.Inf(1)},
		{"-Inf", math.Inf(-1)},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			series := []float64{1, 2, 1, 2, 1, 2, 1, 2, 1, 2}
			series[3] = tc.bad
			model, err := NewARIMA(1, 0, 0)
			require.NoError(t, err)
			assert.Error(t, model.Fit(series))
		})
	}
}

func TestFitTooShort(t *testing.T) {
	model, err := NewARIMA(2, 1, 0)
	require.NoError(t, err)
	assert.Error(t, model.Fit([]float64{1, 2}))
}

func TestForecastBeforeFitErrors(t *testing.T) {
	// Forecasting an unfitted model must error rather than silently returning a
	// plausible-looking all-zero forecast from uninitialized state.
	model, err := NewARIMA(2, 1, 1)
	require.NoError(t, err)
	_, err = model.Forecast(3)
	assert.Error(t, err)
}

func TestForecastInvalidHorizon(t *testing.T) {
	model, err := NewARIMA(1, 0, 0)
	require.NoError(t, err)
	require.NoError(t, model.Fit([]float64{1, 2, 1, 2, 1, 2}))
	_, err = model.Forecast(0)
	assert.Error(t, err)
}

func TestGetters(t *testing.T) {
	data := []float64{1, 2, 1, 2, 1, 2, 1, 2, 1, 2}
	model, err := NewARIMA(1, 0, 0)
	require.NoError(t, err)
	require.NoError(t, model.Fit(data))

	p, d, q := model.Orders()
	assert.Equal(t, [3]int{1, 0, 0}, [3]int{p, d, q})
	assert.Len(t, model.Phi(), 1)
	assert.Empty(t, model.Theta())
	assert.Len(t, model.LastY(), 1)
	assert.Empty(t, model.LastE())
	assert.Equal(t, 2.0, model.LastOrig())
	assert.GreaterOrEqual(t, model.Sigma2(), 0.0)
}

func TestGettersReturnCopies(t *testing.T) {
	// The slice getters must return copies: mutating a returned slice must not
	// corrupt the model's state or change its forecasts.
	series := genARMA11(500, 0.5, 0.4, 17)
	model, err := NewARIMA(1, 0, 1)
	require.NoError(t, err)
	require.NoError(t, model.Fit(series))

	before, err := model.Forecast(3)
	require.NoError(t, err)

	model.Phi()[0] = 99
	model.Theta()[0] = 99
	model.LastY()[0] = 99
	model.LastE()[0] = 99

	assert.NotEqual(t, 99.0, model.Phi()[0])
	assert.NotEqual(t, 99.0, model.Theta()[0])
	assert.NotEqual(t, 99.0, model.LastY()[0])
	assert.NotEqual(t, 99.0, model.LastE()[0])

	after, err := model.Forecast(3)
	require.NoError(t, err)
	assert.Equal(t, before, after)
}

func TestFitWithCSSRefinementLowersResidualVariance(t *testing.T) {
	// On real ARMA data, refining the HR estimate by CSS cannot increase the
	// residual variance (it improves or falls back to the seed).
	series := genARMA11(3000, 0.5, 0.4, 3)

	plain, err := NewARIMA(1, 0, 1)
	require.NoError(t, err)
	require.NoError(t, plain.Fit(series))

	refined, err := NewARIMA(1, 0, 1)
	require.NoError(t, err)
	require.NoError(t, refined.Fit(series, WithCSSRefinement()))

	assert.LessOrEqual(t, refined.Sigma2(), plain.Sigma2())
}

func TestFitWithCSSRefinementRandomWalkNoop(t *testing.T) {
	// ARIMA(0,1,0) has no coefficients to refine; the option must be a harmless
	// no-op rather than an error.
	series := []float64{1, 3, 2, 5, 4, 7, 6, 9, 8, 11}
	model, err := NewARIMA(0, 1, 0)
	require.NoError(t, err)
	require.NoError(t, model.Fit(series, WithCSSRefinement()))
	assert.Empty(t, model.Phi())
	assert.Empty(t, model.Theta())
}

func TestFitWithMLEChangesCoefficients(t *testing.T) {
	// The MLE option must actually refine away from the Hannan-Rissanen seed,
	// not silently no-op.
	series := genARMA11(2000, 0.5, 0.4, 13)

	plain, err := NewARIMA(1, 0, 1)
	require.NoError(t, err)
	require.NoError(t, plain.Fit(series))

	mle, err := NewARIMA(1, 0, 1)
	require.NoError(t, err)
	require.NoError(t, mle.Fit(series, WithMLE()))

	assert.NotEqual(t, plain.Phi()[0], mle.Phi()[0])
	assert.True(t, isStationary(mle.Phi()))
	assert.True(t, isInvertible(mle.Theta()))
}

func TestFitWithMLEFiniteForecast(t *testing.T) {
	// An MLE-refined fit must still produce a finite forecast.
	series := genARMA11(1000, 0.6, 0.3, 21)
	model, err := NewARIMA(1, 0, 1)
	require.NoError(t, err)
	require.NoError(t, model.Fit(series, WithMLE()))

	forecast, err := model.Forecast(5)
	require.NoError(t, err)
	for _, f := range forecast {
		assert.False(t, math.IsNaN(f) || math.IsInf(f, 0))
	}
}

func TestFitWithMLETakesPrecedenceOverCSS(t *testing.T) {
	// When both options are supplied, MLE wins (matching modern statsmodels'
	// statespace default), so the fit equals an MLE-only fit.
	series := genARMA11(2000, 0.5, 0.4, 11)

	both, err := NewARIMA(1, 0, 1)
	require.NoError(t, err)
	require.NoError(t, both.Fit(series, WithCSSRefinement(), WithMLE()))

	mleOnly, err := NewARIMA(1, 0, 1)
	require.NoError(t, err)
	require.NoError(t, mleOnly.Fit(series, WithMLE()))

	assert.Equal(t, mleOnly.Phi(), both.Phi())
	assert.Equal(t, mleOnly.Theta(), both.Theta())
}

func TestFitWithMLERandomWalkNoop(t *testing.T) {
	// ARIMA(0,1,0) has no coefficients to refine; the option must be a harmless
	// no-op rather than an error.
	series := []float64{1, 3, 2, 5, 4, 7, 6, 9, 8, 11}
	model, err := NewARIMA(0, 1, 0)
	require.NoError(t, err)
	require.NoError(t, model.Fit(series, WithMLE()))
	assert.Empty(t, model.Phi())
	assert.Empty(t, model.Theta())
}

func TestARIMA(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	p, d, q := 1, 1, 1

	model, err := NewARIMA(p, d, q)
	require.NoError(t, err)
	require.NotNil(t, model)

	err = model.Fit(data)
	require.NoError(t, err)

	forecast, err := model.Forecast(5)
	require.NoError(t, err)
	assert.Equal(t, 5, len(forecast))
	assert.Equal(t, 11.0, forecast[0])
	assert.Equal(t, 12.0, forecast[1])
	assert.Equal(t, 13.0, forecast[2])
	assert.Equal(t, 14.0, forecast[3])
	assert.Equal(t, 15.0, forecast[4])
}
