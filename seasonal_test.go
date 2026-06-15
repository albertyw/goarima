package goarima

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

// strongSeasonal returns periods*m points of a fixed seasonal shape plus small
// noise, so its seasonal strength is high.
func strongSeasonal(periods, m int, amp, noise float64, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	pattern := make([]float64, m)
	for s := range pattern {
		pattern[s] = amp * math.Sin(2*math.Pi*float64(s)/float64(m))
	}
	out := make([]float64, periods*m)
	for i := range out {
		out[i] = pattern[i%m] + noise*r.NormFloat64()
	}
	return out
}

func TestSelectSeasonalDDetectsStrongSeason(t *testing.T) {
	series := strongSeasonal(12, 12, 10, 0.3, 1)
	assert.Equal(t, 1, selectSeasonalD(series, 12))
}

func TestSelectSeasonalDIgnoresNoise(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	series := make([]float64, 144)
	for i := range series {
		series[i] = r.NormFloat64()
	}
	assert.Equal(t, 0, selectSeasonalD(series, 12))
}

func TestSelectSeasonalDShortSeriesReturnsZero(t *testing.T) {
	assert.Equal(t, 0, selectSeasonalD([]float64{1, 2, 3, 4, 5}, 12))
}

func TestSelectSeasonalDConstantReturnsZero(t *testing.T) {
	series := make([]float64, 60)
	for i := range series {
		series[i] = 7
	}
	assert.Equal(t, 0, selectSeasonalD(series, 12))
}

func TestSeasonalStrengthBoundedZeroOne(t *testing.T) {
	series := strongSeasonal(10, 6, 5, 1, 3)
	fs := seasonalStrength(series, 6)
	assert.GreaterOrEqual(t, fs, 0.0)
	assert.LessOrEqual(t, fs, 1.0)
}

func TestCenteredMovingAverageOddAndEven(t *testing.T) {
	// On a pure linear trend the centered MA equals the series where defined.
	series := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	for _, m := range []int{3, 4} {
		trend := centeredMovingAverage(series, m)
		for i := m / 2; i < len(series)-m/2; i++ {
			assert.InDelta(t, series[i], trend[i], 1e-9)
		}
	}
}
