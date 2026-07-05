package goarima

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRandomWalkFits(t *testing.T) {
	// ARIMA(0,1,0) is the random-walk(-with-drift) baseline: no AR or MA terms,
	// just differencing. It must fit and forecast, not error out.
	series := []float64{1, 3, 2, 5, 4, 7, 6, 9, 8, 11}
	model, err := NewARIMA(Order{P: 0, D: 1, Q: 0})
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
	model, err := NewARIMA(Order{P: 0, D: 1, Q: 1})
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
			model, err := NewARIMA(Order{P: tc.p, D: tc.d, Q: tc.q})
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
	_, err := NewARIMA(Order{P: -1, D: 0, Q: 0})
	assert.Error(t, err)
	_, err = NewARIMA(Order{P: 0, D: 0, Q: 0})
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
			model, err := NewARIMA(Order{P: 1, D: 0, Q: 0})
			require.NoError(t, err)
			assert.Error(t, model.Fit(series))
		})
	}
}

func TestFitTooShort(t *testing.T) {
	model, err := NewARIMA(Order{P: 2, D: 1, Q: 0})
	require.NoError(t, err)
	assert.Error(t, model.Fit([]float64{1, 2}))
}

func TestFitRejectsSeriesShorterThanExpandedMA(t *testing.T) {
	// The expanded seasonal MA polynomial has degree q + Q·m (here 0 + 1·12 = 12),
	// longer than the 8-point series. The length guard must account for the MA
	// side too, otherwise the residual tail slice goes negative and panics.
	model, err := NewSARIMA(Order{P: 0, D: 0, Q: 0}, SeasonalOrder{P: 0, D: 0, Q: 1, Period: 12})
	require.NoError(t, err)
	series := make([]float64, 8)
	for i := range series {
		series[i] = math.Sin(float64(i))
	}
	assert.Error(t, model.Fit(series))
}

func TestForecastBeforeFitErrors(t *testing.T) {
	// Forecasting an unfitted model must error rather than silently returning a
	// plausible-looking all-zero forecast from uninitialized state.
	model, err := NewARIMA(Order{P: 2, D: 1, Q: 1})
	require.NoError(t, err)
	_, err = model.Forecast(3)
	assert.Error(t, err)
}

func TestForecastInvalidHorizon(t *testing.T) {
	model, err := NewARIMA(Order{P: 1, D: 0, Q: 0})
	require.NoError(t, err)
	require.NoError(t, model.Fit([]float64{1, 2, 1, 2, 1, 2}))
	_, err = model.Forecast(0)
	assert.Error(t, err)
}

func TestGetters(t *testing.T) {
	data := []float64{1, 2, 1, 2, 1, 2, 1, 2, 1, 2}
	model, err := NewARIMA(Order{P: 1, D: 0, Q: 0})
	require.NoError(t, err)
	require.NoError(t, model.Fit(data))

	assert.Equal(t, Order{P: 1, D: 0, Q: 0}, model.Order())
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
	model, err := NewARIMA(Order{P: 1, D: 0, Q: 1})
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

	plain, err := NewARIMA(Order{P: 1, D: 0, Q: 1})
	require.NoError(t, err)
	require.NoError(t, plain.Fit(series))

	refined, err := NewARIMA(Order{P: 1, D: 0, Q: 1})
	require.NoError(t, err)
	require.NoError(t, refined.Fit(series, WithMethod(CSS)))

	assert.LessOrEqual(t, refined.Sigma2(), plain.Sigma2())
}

func TestFitWithCSSRefinementRandomWalkNoop(t *testing.T) {
	// ARIMA(0,1,0) has no coefficients to refine; the option must be a harmless
	// no-op rather than an error.
	series := []float64{1, 3, 2, 5, 4, 7, 6, 9, 8, 11}
	model, err := NewARIMA(Order{P: 0, D: 1, Q: 0})
	require.NoError(t, err)
	require.NoError(t, model.Fit(series, WithMethod(CSS)))
	assert.Empty(t, model.Phi())
	assert.Empty(t, model.Theta())
}

func TestFitWithMLEChangesCoefficients(t *testing.T) {
	// The MLE option must actually refine away from the Hannan-Rissanen seed,
	// not silently no-op.
	series := genARMA11(2000, 0.5, 0.4, 13)

	plain, err := NewARIMA(Order{P: 1, D: 0, Q: 1})
	require.NoError(t, err)
	require.NoError(t, plain.Fit(series))

	mle, err := NewARIMA(Order{P: 1, D: 0, Q: 1})
	require.NoError(t, err)
	require.NoError(t, mle.Fit(series, WithMethod(MLE)))

	assert.NotEqual(t, plain.Phi()[0], mle.Phi()[0])
	assert.True(t, isStationary(mle.Phi()))
	assert.True(t, isInvertible(mle.Theta()))
}

func TestFitWithMLEFiniteForecast(t *testing.T) {
	// An MLE-refined fit must still produce a finite forecast.
	series := genARMA11(1000, 0.6, 0.3, 21)
	model, err := NewARIMA(Order{P: 1, D: 0, Q: 1})
	require.NoError(t, err)
	require.NoError(t, model.Fit(series, WithMethod(MLE)))

	forecast, err := model.Forecast(5)
	require.NoError(t, err)
	for _, f := range forecast {
		assert.False(t, math.IsNaN(f) || math.IsInf(f, 0))
	}
}

func TestFitLastMethodWins(t *testing.T) {
	// Options apply in order, so a later WithMethod overrides an earlier one:
	// WithMethod(CSS) then WithMethod(MLE) equals an MLE-only fit.
	series := genARMA11(2000, 0.5, 0.4, 11)

	both, err := NewARIMA(Order{P: 1, D: 0, Q: 1})
	require.NoError(t, err)
	require.NoError(t, both.Fit(series, WithMethod(CSS), WithMethod(MLE)))

	mleOnly, err := NewARIMA(Order{P: 1, D: 0, Q: 1})
	require.NoError(t, err)
	require.NoError(t, mleOnly.Fit(series, WithMethod(MLE)))

	assert.Equal(t, mleOnly.Phi(), both.Phi())
	assert.Equal(t, mleOnly.Theta(), both.Theta())
}

func TestFitWithMLERandomWalkNoop(t *testing.T) {
	// ARIMA(0,1,0) has no coefficients to refine; the option must be a harmless
	// no-op rather than an error.
	series := []float64{1, 3, 2, 5, 4, 7, 6, 9, 8, 11}
	model, err := NewARIMA(Order{P: 0, D: 1, Q: 0})
	require.NoError(t, err)
	require.NoError(t, model.Fit(series, WithMethod(MLE)))
	assert.Empty(t, model.Phi())
	assert.Empty(t, model.Theta())
}

func TestARIMA(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	p, d, q := 1, 1, 1

	model, err := NewARIMA(Order{P: p, D: d, Q: q})
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

func TestNewSARIMARejectsSeasonalPeriodBelowTwo(t *testing.T) {
	_, err := NewSARIMA(Order{P: 1, D: 0, Q: 0}, SeasonalOrder{P: 0, D: 1, Q: 0, Period: 1}) // D>0 but m<2
	require.Error(t, err)
}

func TestNewSARIMARejectsAllZeroOrders(t *testing.T) {
	_, err := NewSARIMA(Order{P: 0, D: 0, Q: 0}, SeasonalOrder{P: 0, D: 0, Q: 0, Period: 0})
	require.Error(t, err)
}

func TestNewARIMAStillWorks(t *testing.T) {
	m, err := NewARIMA(Order{P: 1, D: 1, Q: 0})
	require.NoError(t, err)
	assert.Equal(t, Order{P: 1, D: 1, Q: 0}, m.Order())
	assert.Equal(t, SeasonalOrder{}, m.SeasonalOrder())
}

func TestSeasonalRandomWalkForecastRepeatsSeason(t *testing.T) {
	// A pure seasonal random walk x_t = x_{t-m} + e: with (0,0,0)(0,1,0)_m the
	// h-step forecast equals the last observed full season, repeated.
	m := 4
	r := rand.New(rand.NewSource(7))
	x := []float64{10, 20, 30, 40} // first season
	for i := 4; i < 40; i++ {
		x = append(x, x[i-m]+0.001*r.NormFloat64())
	}
	model, err := NewSARIMA(Order{P: 0, D: 0, Q: 0}, SeasonalOrder{P: 0, D: 1, Q: 0, Period: m})
	require.NoError(t, err)
	require.NoError(t, model.Fit(x))
	fc, err := model.Forecast(m)
	require.NoError(t, err)
	for i := 0; i < m; i++ {
		assert.InDelta(t, x[len(x)-m+i], fc[i], 0.5)
	}
}

func TestCombinedRegularAndSeasonalForecastFinite(t *testing.T) {
	m := 12
	series := make([]float64, 0, 60)
	for i := 0; i < 60; i++ {
		series = append(series, float64(i)+10*float64(i%m))
	}
	model, err := NewSARIMA(Order{P: 1, D: 1, Q: 0}, SeasonalOrder{P: 0, D: 1, Q: 0, Period: m})
	require.NoError(t, err)
	require.NoError(t, model.Fit(series))
	fc, err := model.Forecast(12)
	require.NoError(t, err)
	for _, v := range fc {
		assert.False(t, math.IsNaN(v) || math.IsInf(v, 0))
	}
}
