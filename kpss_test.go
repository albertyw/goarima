package goarima

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestKPSSLevelStationary(t *testing.T) {
	testCases := []struct {
		name   string
		series []float64
		want   bool
	}{
		{"white noise is stationary", genAR1(500, 0, 1), true},
		{"stationary AR(1)", genAR1(500, 0.6, 2), true},
		{"random walk is non-stationary", randomWalk(500, 3), false},
		{"linear trend is non-stationary", rampWithNoise(500, 0.5, 4), false},
		{"constant is stationary", make([]float64, 50), true},
		{"too short defaults to stationary", []float64{1, 2}, true},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, kpssLevelStationary(tc.series))
		})
	}
}
