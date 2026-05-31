package goarima

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// genAR1 returns a centered AR(1) series z_t = phi*z_{t-1} + e_t.
func genAR1(n int, phi float64, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	burn := 200
	z := make([]float64, n+burn)
	for t := 1; t < len(z); t++ {
		z[t] = phi*z[t-1] + r.NormFloat64()
	}
	return z[burn:]
}

// genMA1 returns a centered MA(1) series z_t = e_t + theta*e_{t-1}.
func genMA1(n int, theta float64, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	burn := 200
	z := make([]float64, n+burn)
	prevE := r.NormFloat64()
	for t := 0; t < len(z); t++ {
		e := r.NormFloat64()
		z[t] = e + theta*prevE
		prevE = e
	}
	return z[burn:]
}

// genARMA11 returns a centered ARMA(1,1) series
// z_t = phi*z_{t-1} + e_t + theta*e_{t-1}.
func genARMA11(n int, phi, theta float64, seed int64) []float64 {
	r := rand.New(rand.NewSource(seed))
	burn := 200
	z := make([]float64, n+burn)
	prevE := r.NormFloat64()
	for t := 1; t < len(z); t++ {
		e := r.NormFloat64()
		z[t] = phi*z[t-1] + e + theta*prevE
		prevE = e
	}
	return z[burn:]
}

func TestHannanRissanenPureAR(t *testing.T) {
	z := genAR1(2000, 0.6, 1)
	phi, theta, residuals, err := hannanRissanen(z, 1, 0)
	require.NoError(t, err)
	assert.Empty(t, theta)
	assert.Len(t, residuals, len(z))
	assert.InDelta(t, 0.6, phi[0], 0.1)
}

func TestHannanRissanenPureMA(t *testing.T) {
	z := genMA1(2000, 0.5, 2)
	phi, theta, _, err := hannanRissanen(z, 0, 1)
	require.NoError(t, err)
	assert.Empty(t, phi)
	assert.InDelta(t, 0.5, theta[0], 0.1)
}

func TestHannanRissanenARMA(t *testing.T) {
	z := genARMA11(3000, 0.5, 0.4, 3)
	phi, theta, _, err := hannanRissanen(z, 1, 1)
	require.NoError(t, err)
	assert.InDelta(t, 0.5, phi[0], 0.1)
	assert.InDelta(t, 0.4, theta[0], 0.1)
}

func TestHannanRissanenConstant(t *testing.T) {
	z := make([]float64, 50) // all zeros (a centered constant series)
	phi, theta, residuals, err := hannanRissanen(z, 1, 1)
	require.NoError(t, err)
	assert.Equal(t, []float64{0}, phi)
	assert.Equal(t, []float64{0}, theta)
	assert.Len(t, residuals, 50)
}

func TestHannanRissanenTooShort(t *testing.T) {
	z := genAR1(10, 0.5, 4)
	_, _, _, err := hannanRissanen(z, 2, 2)
	assert.Error(t, err)
}

func TestArmaResiduals(t *testing.T) {
	// Pure AR(1) with phi=0.5: e_t = z_t - 0.5*z_{t-1}, e_0 = z_0.
	z := []float64{2, 3, 4}
	got := armaResiduals(z, []float64{0.5}, nil)
	assert.InDeltaSlice(t, []float64{2, 2, 2.5}, got, 1e-9)
}

func TestArmaResidualsWithMA(t *testing.T) {
	// ARMA(1,1): e_t = z_t - phi*z_{t-1} - theta*e_{t-1}.
	z := []float64{1, 2, 3}
	phi := []float64{0.5}
	theta := []float64{0.2}
	// e0 = 1
	// e1 = 2 - 0.5*1 - 0.2*1 = 1.3
	// e2 = 3 - 0.5*2 - 0.2*1.3 = 1.74
	got := armaResiduals(z, phi, theta)
	assert.InDeltaSlice(t, []float64{1, 1.3, 1.74}, got, 1e-9)
}

func TestHrAROrder(t *testing.T) {
	assert.GreaterOrEqual(t, hrAROrder(2000, 1, 1), 3) // at least p+q+1
	assert.LessOrEqual(t, hrAROrder(20, 1, 1), 10)     // bounded by n/2
}
