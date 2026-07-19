package goarima

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/albertyw/gaussian"
	"gonum.org/v1/gonum/stat/distuv"
)

// numHessian returns the symmetric Hessian of f at x, approximated by central
// second-order finite differences. The per-coordinate step is scaled to each
// argument's magnitude so it works across parameter scales.
func numHessian(f func([]float64) float64, x []float64) [][]float64 {
	n := len(x)
	h := make([]float64, n)
	for i, xi := range x {
		h[i] = 1e-4 * math.Max(1, math.Abs(xi))
	}
	step := func(deltas map[int]float64) []float64 {
		y := append([]float64(nil), x...)
		for i, s := range deltas {
			y[i] += s
		}
		return y
	}
	f0 := f(x)
	H := make([][]float64, n)
	for i := range H {
		H[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		fp := f(step(map[int]float64{i: h[i]}))
		fm := f(step(map[int]float64{i: -h[i]}))
		H[i][i] = (fp - 2*f0 + fm) / (h[i] * h[i])
		for j := i + 1; j < n; j++ {
			fpp := f(step(map[int]float64{i: h[i], j: h[j]}))
			fpm := f(step(map[int]float64{i: h[i], j: -h[j]}))
			fmp := f(step(map[int]float64{i: -h[i], j: h[j]}))
			fmm := f(step(map[int]float64{i: -h[i], j: -h[j]}))
			H[i][j] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
			H[j][i] = H[i][j]
		}
	}
	return H
}

// packCoef stacks the fitted coefficients into the canonical order
// (β, φ, Φₛ, θ, Θₛ) used by paramNames, StdErrors, and the NLL objectives.
func packCoef(beta, phi, sphi, theta, stheta []float64) []float64 {
	out := make([]float64, 0, len(beta)+len(phi)+len(sphi)+len(theta)+len(stheta))
	out = append(out, beta...)
	out = append(out, phi...)
	out = append(out, sphi...)
	out = append(out, theta...)
	out = append(out, stheta...)
	return out
}

// paramNames returns the canonical coefficient names (excluding sigma2) in the
// order (β, φ, Φₛ, θ, Θₛ), mirroring statsmodels' labels.
func paramNames(exogDim, p, bigP, q, bigQ, period int) []string {
	var names []string
	for i := 1; i <= exogDim; i++ {
		names = append(names, fmt.Sprintf("x%d", i))
	}
	for i := 1; i <= p; i++ {
		names = append(names, fmt.Sprintf("ar.L%d", i))
	}
	for i := 1; i <= bigP; i++ {
		names = append(names, fmt.Sprintf("ar.S.L%d", i*period))
	}
	for i := 1; i <= q; i++ {
		names = append(names, fmt.Sprintf("ma.L%d", i))
	}
	for i := 1; i <= bigQ; i++ {
		names = append(names, fmt.Sprintf("ma.S.L%d", i*period))
	}
	return names
}

// armaNLLObjective evaluates the concentrated NLL as a function of the packed
// factor vector (φ, Φₛ, θ, Θₛ) on the fixed centered series z — the objective
// refineSeasonalMLE minimizes, without the stability penalty (the fitted point
// is strictly inside the stable region; kalmanConcentratedNLL still guards).
func armaNLLObjective(z []float64, p, bigP, q, period int) func([]float64) float64 {
	return func(x []float64) float64 {
		phi, sphi := x[:p], x[p:p+bigP]
		theta, stheta := x[p+bigP:p+bigP+q], x[p+bigP+q:]
		return kalmanConcentratedNLL(z, expandSeasonalAR(phi, sphi, period), expandSeasonalMA(theta, stheta, period))
	}
}

// exogNLLObjective evaluates the concentrated NLL as a function of the packed
// vector (β, φ, Φₛ, θ, Θₛ), re-forming η = y − Xβ and re-centering each call —
// the objective refineExog minimizes.
func exogNLLObjective(series []float64, X [][]float64, k, p, bigP, q, d, bigD, period int) func([]float64) float64 {
	return func(x []float64) float64 {
		beta := x[:k]
		phi, sphi := x[k:k+p], x[k+p:k+p+bigP]
		theta, stheta := x[k+p+bigP:k+p+bigP+q], x[k+p+bigP+q:]
		z := centeredDiff(regressionResiduals(series, X, beta), d, bigD, period)
		return kalmanConcentratedNLL(z, expandSeasonalAR(phi, sphi, period), expandSeasonalMA(theta, stheta, period))
	}
}

// paramCovariance returns cov = 2·H⁻¹ for the fitted coefficient vector coef,
// where H is the numeric Hessian of objective at coef. The factor of 2 converts
// the objective (−2·loglik + const) into the observed information. It returns
// nil when H is singular (standard errors unavailable). H is cloned per solved
// column because gaussian.Solve works in place.
func paramCovariance(objective func([]float64) float64, coef []float64) [][]float64 {
	n := len(coef)
	if n == 0 {
		return [][]float64{}
	}
	H := numHessian(objective, coef)
	cov := make([][]float64, n)
	for i := range cov {
		cov[i] = make([]float64, n)
	}
	for j := 0; j < n; j++ {
		e := make([]float64, n)
		e[j] = 1
		col, err := gaussian.Solve(cloneMatrix(H), e)
		if err != nil {
			return nil
		}
		for i := 0; i < n; i++ {
			cov[i][j] = 2 * col[i]
		}
	}
	return cov
}

func cloneMatrix(m [][]float64) [][]float64 {
	out := make([][]float64, len(m))
	for i := range m {
		out[i] = append([]float64(nil), m[i]...)
	}
	return out
}

// StdErrors returns the standard errors of the fitted coefficients
// (β, φ, Φₛ, θ, Θₛ) in canonical order. It requires an MLE fit
// (WithMethod(MLE)) and returns an error otherwise, or if the information
// matrix was singular. An entry is NaN when its estimated variance is
// non-positive (a weakly identified coefficient).
func (m *ARIMA) StdErrors() ([]float64, error) {
	if !m.mleFit {
		return nil, errors.New("standard errors require an MLE fit (WithMethod(MLE))")
	}
	if m.paramCov == nil {
		return nil, errors.New("standard errors unavailable: singular information matrix")
	}
	se := make([]float64, len(m.paramCov))
	for i := range m.paramCov {
		se[i] = math.Sqrt(m.paramCov[i][i])
	}
	return se, nil
}

// ParamStat is one row of a Summary: a coefficient with its inference stats.
type ParamStat struct {
	Name    string
	Coef    float64
	StdErr  float64
	ZScore  float64 // Coef / StdErr
	PValue  float64 // two-sided, standard normal
	CILower float64 // 95% by default
	CIUpper float64
}

// Summary bundles a fitted (MLE) model's parameter statistics and fit metrics.
type Summary struct {
	Params []ParamStat // one row per coefficient, then a final sigma2 row
	NObs   int
	LogLik float64
	AIC    float64
	BIC    float64
	Sigma2 float64
}

// Summary returns the parameter-inference summary for an MLE-fit model
// (coefficient estimates, standard errors, z-statistics, p-values, and
// confidence intervals, plus model-level fit statistics). It returns an error
// if the model was not fit with WithMethod(MLE).
func (m *ARIMA) Summary() (*Summary, error) {
	se, err := m.StdErrors()
	if err != nil {
		return nil, err
	}
	coef := packCoef(m.beta, m.phi, m.seasonalPhi, m.theta, m.seasonalTheta)
	z975 := distuv.UnitNormal.Quantile(0.975)
	rows := make([]ParamStat, 0, len(coef)+1)
	for i := range coef {
		rows = append(rows, statRow(m.paramNames[i], coef[i], se[i], z975))
	}
	// sigma2: asymptotic SE σ̂²·√(2/n) (see spec D3).
	rows = append(rows, statRow("sigma2", m.sigma2, m.sigma2*math.Sqrt(2/float64(m.nobs)), z975))

	k := float64(m.exogDim + m.p + m.bigP + m.q + m.bigQ + 1) // +1 = sigma2
	return &Summary{
		Params: rows,
		NObs:   m.nobs,
		LogLik: m.logLik,
		AIC:    -2*m.logLik + 2*k,
		BIC:    -2*m.logLik + k*math.Log(float64(m.nobs)),
		Sigma2: m.sigma2,
	}, nil
}

func statRow(name string, coef, se, z975 float64) ParamStat {
	z := coef / se
	return ParamStat{
		Name:    name,
		Coef:    coef,
		StdErr:  se,
		ZScore:  z,
		PValue:  2 * (1 - distuv.UnitNormal.CDF(math.Abs(z))),
		CILower: coef - z975*se,
		CIUpper: coef + z975*se,
	}
}

// String renders the Summary as a fixed-width table (statsmodels-like).
func (s *Summary) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "n=%d  logLik=%.4f  AIC=%.4f  BIC=%.4f  sigma2=%.6g\n",
		s.NObs, s.LogLik, s.AIC, s.BIC, s.Sigma2)
	fmt.Fprintf(&b, "%-10s %12s %12s %9s %9s %12s %12s\n",
		"param", "coef", "std err", "z", "P>|z|", "[0.025", "0.975]")
	for _, p := range s.Params {
		fmt.Fprintf(&b, "%-10s %12.6g %12.6g %9.3f %9.3f %12.6g %12.6g\n",
			p.Name, p.Coef, p.StdErr, p.ZScore, p.PValue, p.CILower, p.CIUpper)
	}
	return b.String()
}
