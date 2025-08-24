package goarima

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSimpleARIMA(t *testing.T) {
	var testCases = []struct {
		name     string
		data     []float64
		p, d, q  int
		expected []float64
	}{
		{
			name:     "ARIMA(1,0,0) with oscillating data",
			data:     []float64{1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
			p:        1,
			d:        0,
			q:        0,
			expected: []float64{1, 2, 1, 2, 1},
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
			assert.Equal(t, len(forecast), 5)
			for i := range forecast {
				assert.InDelta(t, forecast[i], tc.expected[i], 1e-6)
			}
		})
	}
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
	assert.Equal(t, len(forecast), 5)
	assert.Equal(t, forecast[0], 11.0)
	assert.Equal(t, forecast[1], 12.0)
	assert.Equal(t, forecast[2], 13.0)
	assert.Equal(t, forecast[3], 14.0)
	assert.Equal(t, forecast[4], 15.0)
}
