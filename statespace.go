package goarima

import (
	"math"

	"github.com/albertyw/gaussian"
)

// stateSpace is the Harvey state-space representation of an ARMA(p,q) model used
// by the Kalman filter. The observation vector Z is always [1, 0, …, 0], so it
// is left implicit (the observation is the first state component).
//
//	state:       a_{t+1} = T·a_t + R·η_{t+1},  η ~ N(0, σ²)
//	observation: y_t     = Z·a_t = a_t[0]
//
// with r = max(p, q+1), T the companion transition (phi in the first column, an
// identity shift on the superdiagonal) and R the selection vector [1, θ_1, …].
type stateSpace struct {
	r int         // state dimension, max(p, q+1)
	T [][]float64 // r×r transition matrix
	R []float64   // r selection vector
}

// buildStateSpace constructs the Harvey state-space form for the given AR (phi)
// and MA (theta) coefficients.
func buildStateSpace(phi, theta []float64) stateSpace {
	p := len(phi)
	q := len(theta)
	r := p
	if q+1 > r {
		r = q + 1
	}

	T := make([][]float64, r)
	for i := range T {
		T[i] = make([]float64, r)
		if i < p {
			T[i][0] = phi[i]
		}
		if i+1 < r {
			T[i][i+1] = 1
		}
	}

	R := make([]float64, r)
	R[0] = 1
	for j := 0; j < q; j++ {
		if j+1 < r {
			R[j+1] = theta[j]
		}
	}

	return stateSpace{r: r, T: T, R: R}
}

// solveLyapunov returns the stationary state covariance P (in units of σ²)
// solving the discrete Lyapunov equation
//
//	P = T·P·Tᵀ + R·Rᵀ
//
// by writing the r² linear equations directly (one per entry of P) and solving
// them with gaussian.Solve. For non-stationary parameters the system can yield a
// non-positive variance, which the caller detects via F_t ≤ 0.
func solveLyapunov(ss stateSpace) ([][]float64, error) {
	r := ss.r
	n := r * r
	M := make([][]float64, n)
	rhs := make([]float64, n)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			e := i*r + j
			M[e] = make([]float64, n)
			for k := 0; k < r; k++ {
				for l := 0; l < r; l++ {
					M[e][k*r+l] = -ss.T[i][k] * ss.T[j][l]
				}
			}
			M[e][e]++ // the identity term: P[i][j] coefficient
			rhs[e] = ss.R[i] * ss.R[j]
		}
	}

	sol, err := gaussian.Solve(M, rhs)
	if err != nil {
		return nil, err
	}

	P := make([][]float64, r)
	for i := 0; i < r; i++ {
		P[i] = make([]float64, r)
		for j := 0; j < r; j++ {
			P[i][j] = sol[i*r+j]
		}
	}
	return P, nil
}

// kalmanConcentratedNLL runs the Kalman filter prediction-error decomposition
// for ARMA(phi, theta) on the series y and returns the concentrated negative
// log-likelihood (dropping additive constants):
//
//	σ²̂ = (1/n)·Σ v_t²/F_t,   objective = n·ln(σ²̂) + Σ ln(F_t)
//
// where v_t is the one-step prediction error and F_t its variance (in units of
// σ²). The filter starts from the stationary covariance (solveLyapunov) and a
// zero state mean. It returns +Inf when the parameters are non-stationary (the
// stationary covariance has a non-positive prediction variance), steering the
// optimizer back into the stable region.
func kalmanConcentratedNLL(y, phi, theta []float64) float64 {
	n := len(y)
	ss := buildStateSpace(phi, theta)
	r := ss.r

	P, err := solveLyapunov(ss)
	if err != nil {
		return math.Inf(1)
	}

	a := make([]float64, r) // state mean, a_1 = 0
	var sumLogF, sumVsq float64

	for t := 0; t < n; t++ {
		// Prediction error and its variance (Z picks the first component).
		v := y[t] - a[0]
		f := P[0][0]
		if f <= 0 || math.IsNaN(f) {
			return math.Inf(1)
		}
		sumLogF += math.Log(f)
		sumVsq += v * v / f

		// TP = T·P, then TPZt = first column of TP = T·P·Zᵀ.
		tp := make([][]float64, r)
		tpzt := make([]float64, r)
		for i := 0; i < r; i++ {
			tp[i] = make([]float64, r)
			for j := 0; j < r; j++ {
				var s float64
				for k := 0; k < r; k++ {
					s += ss.T[i][k] * P[k][j]
				}
				tp[i][j] = s
			}
			tpzt[i] = tp[i][0]
		}

		// State update: a_{t+1} = T·a_t + K·v, with K = TPZt / F.
		newA := make([]float64, r)
		for i := 0; i < r; i++ {
			var ta float64
			for k := 0; k < r; k++ {
				ta += ss.T[i][k] * a[k]
			}
			newA[i] = ta + tpzt[i]/f*v
		}
		a = newA

		// Covariance update: P_{t+1} = T·P·Tᵀ − (TPZt)(TPZt)ᵀ/F + R·Rᵀ.
		newP := make([][]float64, r)
		for i := 0; i < r; i++ {
			newP[i] = make([]float64, r)
			for j := 0; j < r; j++ {
				var tptt float64
				for k := 0; k < r; k++ {
					tptt += tp[i][k] * ss.T[j][k]
				}
				newP[i][j] = tptt - tpzt[i]*tpzt[j]/f + ss.R[i]*ss.R[j]
			}
		}
		P = newP
	}

	return float64(n)*math.Log(sumVsq/float64(n)) + sumLogF
}
