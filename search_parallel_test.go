package goarima

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWithParallelSetsConfig(t *testing.T) {
	var cfg fitConfig
	assert.False(t, cfg.parallel, "default is serial")
	WithParallel()(&cfg)
	assert.True(t, cfg.parallel)
}

// sameModel asserts two fitted models selected the same order and coefficients.
func sameModel(t *testing.T, want, got *ARIMA) {
	t.Helper()
	wp, wd, wq := want.Orders()
	gp, gd, gq := got.Orders()
	assert.Equal(t, [3]int{wp, wd, wq}, [3]int{gp, gd, gq}, "orders differ")
	assert.Equal(t, want.Phi(), got.Phi(), "phi differs")
	assert.Equal(t, want.Theta(), got.Theta(), "theta differs")
}

func TestParallelGridMatchesSerial(t *testing.T) {
	for _, series := range [][]float64{
		ar1Series(300, 0.7, 1),
		ar1Series(250, -0.4, 2),
		rampWithNoise(220, 0.5, 3),
	} {
		serial, err := AutoARIMA(series, 4, 2, 4)
		require.NoError(t, err)
		par, err := AutoARIMA(series, 4, 2, 4, WithParallel())
		require.NoError(t, err)
		sameModel(t, serial, par)
	}
}

func TestParallelStepwiseMatchesSerial(t *testing.T) {
	for _, series := range [][]float64{
		ar1Series(300, 0.6, 4),
		rampWithNoise(240, 0.3, 5),
	} {
		serial, err := AutoARIMA(series, 5, 2, 5, WithStepwise())
		require.NoError(t, err)
		par, err := AutoARIMA(series, 5, 2, 5, WithStepwise(), WithParallel())
		require.NoError(t, err)
		sameModel(t, serial, par)
	}
}
