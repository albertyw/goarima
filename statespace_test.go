package goarima

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestBuildStateSpaceAR1 checks the Harvey state-space form of an AR(1): the
// state dimension is 1, the transition is the scalar phi, and the selection
// vector is [1].
func TestBuildStateSpaceAR1(t *testing.T) {
	ss := buildStateSpace([]float64{0.7}, nil)
	assert.Equal(t, 1, ss.r)
	assert.Equal(t, [][]float64{{0.7}}, ss.T)
	assert.Equal(t, []float64{1}, ss.R)
}

// TestBuildStateSpaceMA1 checks the Harvey form of an MA(1): r = q+1 = 2, the
// companion transition has a superdiagonal 1, and the selection vector carries
// the MA coefficient.
func TestBuildStateSpaceMA1(t *testing.T) {
	ss := buildStateSpace(nil, []float64{0.4})
	assert.Equal(t, 2, ss.r)
	assert.Equal(t, [][]float64{{0, 1}, {0, 0}}, ss.T)
	assert.Equal(t, []float64{1, 0.4}, ss.R)
}

// TestBuildStateSpaceARMA21 checks an ARMA(2,1): r = max(p, q+1) = 2, phi fills
// the first column, the superdiagonal is the identity shift, and theta fills the
// selection vector below the leading 1.
func TestBuildStateSpaceARMA21(t *testing.T) {
	ss := buildStateSpace([]float64{0.5, -0.3}, []float64{0.4})
	assert.Equal(t, 2, ss.r)
	assert.Equal(t, [][]float64{{0.5, 1}, {-0.3, 0}}, ss.T)
	assert.Equal(t, []float64{1, 0.4}, ss.R)
}

// TestSolveLyapunovAR1 checks the stationary state covariance of an AR(1): in
// units of sigma^2 it is the familiar 1/(1-phi^2).
func TestSolveLyapunovAR1(t *testing.T) {
	ss := buildStateSpace([]float64{0.5}, nil)
	P, err := solveLyapunov(ss)
	require.NoError(t, err)
	assert.InDelta(t, 1.0/(1-0.25), P[0][0], 1e-12)
}

// TestSolveLyapunovMA1 checks the stationary state covariance of an MA(1)
// against its closed form: P00 = 1+theta^2, P01 = theta, P11 = theta^2.
func TestSolveLyapunovMA1(t *testing.T) {
	ss := buildStateSpace(nil, []float64{0.4})
	P, err := solveLyapunov(ss)
	require.NoError(t, err)
	assert.InDelta(t, 1.16, P[0][0], 1e-12)
	assert.InDelta(t, 0.4, P[0][1], 1e-12)
	assert.InDelta(t, 0.16, P[1][1], 1e-12)
}

// TestKalmanConcentratedNLLAR1MatchesClosedForm validates the whole filter
// pipeline (state-space build, stationary init, prediction-error recursion,
// concentrated likelihood) against the exact AR(1) Gaussian likelihood, whose
// concentrated form with a stationary first observation is known in closed form.
func TestKalmanConcentratedNLLAR1MatchesClosedForm(t *testing.T) {
	y := []float64{1.0, 2.0, 1.5, 3.0}
	phi := 0.5

	got := kalmanConcentratedNLL(y, []float64{phi}, nil)

	n := float64(len(y))
	ss2 := (1 - phi*phi) * y[0] * y[0]
	for i := 1; i < len(y); i++ {
		d := y[i] - phi*y[i-1]
		ss2 += d * d
	}
	ss2 /= n
	// objective = n*ln(sigma2_hat) + sum ln(F_t); for AR(1) sum ln(F_t) = -ln(1-phi^2).
	want := n*math.Log(ss2) - math.Log(1-phi*phi)
	assert.InDelta(t, want, got, 1e-9)
}

// TestKalmanConcentratedNLLNonStationaryInf returns +Inf when the parameters are
// non-stationary, so the Lyapunov solve has no valid stationary covariance and
// the optimizer is steered away from that region.
func TestKalmanConcentratedNLLNonStationaryInf(t *testing.T) {
	y := []float64{1.0, 2.0, 1.5, 3.0}
	got := kalmanConcentratedNLL(y, []float64{1.5}, nil)
	assert.True(t, math.IsInf(got, 1))
}
