package goarima

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsStationary(t *testing.T) {
	testCases := []struct {
		name string
		phi  []float64
		want bool
	}{
		{"empty (no AR) is stationary", nil, true},
		{"AR(1) inside unit circle", []float64{0.5}, true},
		{"AR(1) near boundary", []float64{-0.995}, true},
		{"AR(1) explosive", []float64{1.2}, false},
		{"AR(1) unit root", []float64{1.0}, false},
		{"AR(2) stationary", []float64{0.5, -0.3}, true},
		{"AR(2) stationary with phi1>1", []float64{1.1, -0.4}, true},
		{"AR(2) non-stationary (phi1+phi2>1)", []float64{0.5, 0.6}, false},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, isStationary(tc.phi))
		})
	}
}

func TestIsInvertible(t *testing.T) {
	testCases := []struct {
		name  string
		theta []float64
		want  bool
	}{
		{"empty (no MA) is invertible", nil, true},
		{"MA(1) inside unit circle", []float64{0.4}, true},
		{"MA(1) non-invertible", []float64{-1.73}, false},
		{"MA(1) unit root", []float64{1.0}, false},
		{"MA(2) invertible", []float64{-0.5, 0.2}, true},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, isInvertible(tc.theta))
		})
	}
}
