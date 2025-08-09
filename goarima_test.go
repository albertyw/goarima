package goarima

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
	assert.Equal(t, len(forecast), 5, "Expected 5 forecast values")
}
