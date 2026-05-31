package goarima

import (
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

func TestFitTooShort(t *testing.T) {
	model, err := NewARIMA(2, 1, 0)
	require.NoError(t, err)
	assert.Error(t, model.Fit([]float64{1, 2}))
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
