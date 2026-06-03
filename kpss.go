package goarima

import "math"

// kpssLevelCritical5 is the asymptotic 5% critical value of the KPSS statistic
// under the null of level (constant-mean) stationarity (Kwiatkowski, Phillips,
// Schmidt & Shin, 1992). A statistic above this rejects stationarity.
const kpssLevelCritical5 = 0.463

// kpssLevelStationary reports whether the series is level-stationary by the
// KPSS test at the 5% significance level. The null hypothesis is stationarity;
// a statistic above the critical value rejects it (the series is treated as
// non-stationary and should be differenced).
//
// The statistic is
//
//	η = ( Σ_t S_t² ) / ( n²·σ²_lr )
//
// where S_t is the cumulative sum of the demeaned series and σ²_lr is its
// long-run variance, estimated with a Newey-West (Bartlett-kernel) correction.
func kpssLevelStationary(series []float64) bool {
	n := len(series)
	if n < 3 {
		return true // too short to call non-stationary
	}

	// Demean (level stationarity regresses on a constant), then form the
	// cumulative partial sums S_t and Σ S_t².
	m := mean(series)
	e := make([]float64, n)
	for i, v := range series {
		e[i] = v - m
	}
	var s, sumS2 float64
	for _, ei := range e {
		s += ei
		sumS2 += s * s
	}

	// Long-run variance via Newey-West with a Bartlett kernel. The bandwidth
	// l = floor(4·(n/100)^¼) is the common automatic choice.
	l := int(4 * math.Pow(float64(n)/100.0, 0.25))
	var g0 float64
	for _, ei := range e {
		g0 += ei * ei
	}
	g0 /= float64(n)
	lrv := g0
	for lag := 1; lag <= l; lag++ {
		var g float64
		for t := lag; t < n; t++ {
			g += e[t] * e[t-lag]
		}
		g /= float64(n)
		weight := 1 - float64(lag)/float64(l+1)
		lrv += 2 * weight * g
	}
	if lrv <= 1e-12 {
		return true // constant/degenerate series is stationary
	}

	eta := sumS2 / (float64(n) * float64(n) * lrv)
	return eta <= kpssLevelCritical5
}
