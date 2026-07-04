package goarima

import (
	"encoding/binary"
	"math"
	"testing"
)

// fuzzSeries decodes raw fuzzer bytes into a finite, magnitude-bounded series.
// Every 8 bytes become one float64; non-finite values collapse to 0 and the
// magnitude is clamped so the forecast recursion cannot overflow to ±Inf. The
// NaN/Inf input guard is already unit-tested (see validateFinite); these
// harnesses instead stress the *bounds* — orders and period against series
// length, and the forecast horizon — where index arithmetic could go wrong.
func fuzzSeries(data []byte) []float64 {
	const limit = 1e6
	s := make([]float64, 0, len(data)/8)
	for i := 0; i+8 <= len(data); i += 8 {
		v := math.Float64frombits(binary.LittleEndian.Uint64(data[i:]))
		if math.IsNaN(v) || math.IsInf(v, 0) {
			v = 0
		}
		if v > limit {
			v = limit
		} else if v < -limit {
			v = -limit
		}
		s = append(s, v)
	}
	return s
}

// floatsToBytes is the inverse of fuzzSeries's decode, for building seed corpora.
func floatsToBytes(s []float64) []byte {
	b := make([]byte, 8*len(s))
	for i, v := range s {
		binary.LittleEndian.PutUint64(b[i*8:], math.Float64bits(v))
	}
	return b
}

// boundOrder maps an arbitrary fuzz int into 0..maxV.
func boundOrder(v, maxV int) int {
	v %= maxV + 1
	if v < 0 {
		v += maxV + 1
	}
	return v
}

// checkForecasts asserts a fitted model's forecasters return exactly h finite
// values (or a clean error), never panicking or emitting non-finite output.
func checkForecasts(t *testing.T, m *ARIMA, h int) {
	t.Helper()
	if pred, err := m.Forecast(h); err == nil {
		if len(pred) != h {
			t.Fatalf("Forecast returned %d values, want %d", len(pred), h)
		}
		for i, v := range pred {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("Forecast produced non-finite value %v at %d", v, i)
			}
		}
	}
	if fc, err := m.ForecastInterval(h, 0.95); err == nil {
		if len(fc.Point) != h || len(fc.Lower) != h || len(fc.Upper) != h {
			t.Fatalf("ForecastInterval lengths %d/%d/%d, want %d", len(fc.Point), len(fc.Lower), len(fc.Upper), h)
		}
		for i := range fc.Point {
			if math.IsNaN(fc.Lower[i]) || math.IsInf(fc.Lower[i], 0) ||
				math.IsNaN(fc.Upper[i]) || math.IsInf(fc.Upper[i], 0) {
				t.Fatalf("ForecastInterval produced non-finite bound at %d", i)
			}
			if fc.Lower[i] > fc.Upper[i] {
				t.Fatalf("ForecastInterval lower %v > upper %v at %d", fc.Lower[i], fc.Upper[i], i)
			}
		}
	}
}

// FuzzFitForecast drives NewARIMA/NewSARIMA + Fit + Forecast/ForecastInterval
// with fuzzed orders, seasonal period, and horizon against a fuzzed series,
// asserting the pipeline never panics and successful forecasts stay finite and
// correctly sized. A degenerate seasonal spec falls back to the regular model.
func FuzzFitForecast(f *testing.F) {
	seed := make([]float64, 60)
	for i := range seed {
		seed[i] = math.Sin(float64(i)*0.4) + 0.5*math.Sin(float64(i)*1.3)
	}
	sb := floatsToBytes(seed)
	f.Add(1, 1, 1, 0, 0, 0, 0, 6, sb)  // regular ARIMA(1,1,1)
	f.Add(0, 1, 1, 0, 1, 1, 12, 8, sb) // airline-style SARIMA
	f.Add(2, 0, 0, 0, 0, 0, 0, 3, sb)  // pure AR(2)

	f.Fuzz(func(t *testing.T, p, d, q, bigP, bigD, bigQ, m, h int, data []byte) {
		p, d, q = boundOrder(p, 4), boundOrder(d, 3), boundOrder(q, 4)
		bigP, bigD, bigQ = boundOrder(bigP, 2), boundOrder(bigD, 2), boundOrder(bigQ, 2)
		m = boundOrder(m, 24)
		h = boundOrder(h, 47) + 1 // 1..48
		series := fuzzSeries(data)

		var (
			model *ARIMA
			err   error
		)
		if m >= 2 && (bigP > 0 || bigD > 0 || bigQ > 0) {
			model, err = NewSARIMA(p, d, q, bigP, bigD, bigQ, m)
		} else {
			model, err = NewARIMA(p, d, q)
		}
		if err != nil {
			return
		}
		if err := model.Fit(series); err != nil {
			return
		}
		checkForecasts(t, model, h)
	})
}

// fuzzMatrix reshapes fuzzed bytes into a rows×cols finite matrix, cycling the
// decoded values (or filling zeros when there are none) so any rows/cols pair is
// representable.
func fuzzMatrix(data []byte, rows, cols int) [][]float64 {
	flat := fuzzSeries(data)
	X := make([][]float64, rows)
	for i := range X {
		X[i] = make([]float64, cols)
		for j := range X[i] {
			if len(flat) > 0 {
				X[i][j] = flat[(i*cols+j)%len(flat)]
			}
		}
	}
	return X
}

// checkExogForecasts is checkForecasts for a model fit with exogenous regressors:
// the exog forecasters must never panic and, when they succeed, return h finite,
// correctly-sized values with lower ≤ upper.
func checkExogForecasts(t *testing.T, m *ARIMA, h int, futureX [][]float64) {
	t.Helper()
	if pred, err := m.ForecastExog(h, futureX); err == nil {
		if len(pred) != h {
			t.Fatalf("ForecastExog returned %d values, want %d", len(pred), h)
		}
		for i, v := range pred {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("ForecastExog produced non-finite value %v at %d", v, i)
			}
		}
	}
	if fc, err := m.ForecastIntervalExog(h, 0.95, futureX); err == nil {
		if len(fc.Point) != h || len(fc.Lower) != h || len(fc.Upper) != h {
			t.Fatalf("ForecastIntervalExog lengths %d/%d/%d, want %d", len(fc.Point), len(fc.Lower), len(fc.Upper), h)
		}
		for i := range fc.Point {
			if math.IsNaN(fc.Lower[i]) || math.IsInf(fc.Lower[i], 0) ||
				math.IsNaN(fc.Upper[i]) || math.IsInf(fc.Upper[i], 0) {
				t.Fatalf("ForecastIntervalExog produced non-finite bound at %d", i)
			}
			if fc.Lower[i] > fc.Upper[i] {
				t.Fatalf("ForecastIntervalExog lower %v > upper %v at %d", fc.Lower[i], fc.Upper[i], i)
			}
		}
	}
}

// FuzzExog drives regression-with-ARIMA-errors: WithExog Fit plus ForecastExog/
// ForecastIntervalExog with a fuzzed n×k design matrix and h×k future rows,
// asserting the β estimation, differencing, and forecast plumbing never panic
// and successful exog forecasts stay finite and correctly sized.
func FuzzExog(f *testing.F) {
	seed := make([]float64, 40)
	for i := range seed {
		seed[i] = math.Sin(float64(i) * 0.3)
	}
	yb := floatsToBytes(seed)
	xb := floatsToBytes(seed) // any finite bytes; reshaped by fuzzMatrix
	f.Add(1, 0, 0, 1, 5, yb, xb)
	f.Add(0, 1, 1, 2, 4, yb, xb)

	f.Fuzz(func(t *testing.T, p, d, q, k, h int, ydata, xdata []byte) {
		p, d, q = boundOrder(p, 3), boundOrder(d, 2), boundOrder(q, 3)
		k = boundOrder(k, 2) + 1  // 1..3
		h = boundOrder(h, 23) + 1 // 1..24
		series := fuzzSeries(ydata)
		if len(series) == 0 {
			return // WithExog requires at least one row
		}
		X := fuzzMatrix(xdata, len(series), k)
		futureX := fuzzMatrix(xdata, h, k)

		model, err := NewARIMA(p, d, q)
		if err != nil {
			return
		}
		if err := model.Fit(series, WithExog(X)); err != nil {
			return
		}
		checkExogForecasts(t, model, h, futureX)
	})
}

// FuzzFitRefine drives the optimizer and root-repair refinement paths
// (WithCSSRefinement/WithMLE/WithRootRepair) on a fuzzed series, asserting the
// gonum Nelder-Mead search and root reflection never panic and still yield
// finite, correctly-sized forecasts.
func FuzzFitRefine(f *testing.F) {
	seed := make([]float64, 50)
	for i := range seed {
		seed[i] = math.Sin(float64(i)*0.5) + 0.3*float64(i%5)
	}
	sb := floatsToBytes(seed)
	f.Add(1, 0, 1, 0, 6, sb)
	f.Add(2, 1, 2, 1, 5, sb)
	f.Add(1, 0, 1, 2, 4, sb)

	f.Fuzz(func(t *testing.T, p, d, q, opt, h int, data []byte) {
		p, d, q = boundOrder(p, 3), boundOrder(d, 2), boundOrder(q, 3)
		h = boundOrder(h, 23) + 1 // 1..24
		series := fuzzSeries(data)

		model, err := NewARIMA(p, d, q)
		if err != nil {
			return
		}
		var opts []FitOption
		switch boundOrder(opt, 3) {
		case 0:
			opts = []FitOption{WithCSSRefinement()}
		case 1:
			opts = []FitOption{WithMLE()}
		case 2:
			opts = []FitOption{WithRootRepair()}
		case 3:
			opts = []FitOption{WithMLE(), WithRootRepair()}
		}
		if err := model.Fit(series, opts...); err != nil {
			return
		}
		checkForecasts(t, model, h)
	})
}

// FuzzAutoARIMA drives the order search with fuzzed caps and horizon against a
// fuzzed series, asserting selection + forecasting never panic and produce
// finite, correctly-sized output.
func FuzzAutoARIMA(f *testing.F) {
	seed := make([]float64, 80)
	for i := range seed {
		seed[i] = float64(i)*0.1 + math.Sin(float64(i)*0.5)
	}
	sb := floatsToBytes(seed)
	f.Add(3, 2, 3, 6, sb)
	f.Add(1, 1, 1, 4, sb)

	f.Fuzz(func(t *testing.T, maxP, maxD, maxQ, h int, data []byte) {
		maxP, maxD, maxQ = boundOrder(maxP, 3), boundOrder(maxD, 2), boundOrder(maxQ, 3)
		h = boundOrder(h, 23) + 1 // 1..24
		series := fuzzSeries(data)

		model, err := AutoARIMA(series, maxP, maxD, maxQ)
		if err != nil {
			return
		}
		checkForecasts(t, model, h)
	})
}

// FuzzAutoSARIMA drives the seasonal order search with fuzzed caps, period, and
// horizon. It exercises the seasonal-D selection (seasonalStrength /
// centeredMovingAverage, whose moving-average window is the period) and the 4-D
// (p,q,P,Q) search — including degenerate periods (0/1) — asserting selection and
// forecasting never panic and stay finite and correctly sized.
func FuzzAutoSARIMA(f *testing.F) {
	seed := make([]float64, 96)
	for i := range seed {
		seed[i] = math.Sin(float64(i)*0.5) + float64(i)*0.05
	}
	sb := floatsToBytes(seed)
	f.Add(2, 1, 2, 1, 1, 12, 6, sb)
	f.Add(1, 1, 1, 1, 0, 4, 4, sb)

	f.Fuzz(func(t *testing.T, maxP, maxD, maxQ, maxBigP, maxBigQ, m, h int, data []byte) {
		maxP, maxD, maxQ = boundOrder(maxP, 2), boundOrder(maxD, 2), boundOrder(maxQ, 2)
		maxBigP, maxBigQ = boundOrder(maxBigP, 1), boundOrder(maxBigQ, 1)
		m = boundOrder(m, 12)     // period 0..12, including the degenerate 0/1
		h = boundOrder(h, 23) + 1 // 1..24
		series := fuzzSeries(data)

		model, err := AutoSARIMA(series, maxP, maxD, maxQ, maxBigP, maxBigQ, m)
		if err != nil {
			return
		}
		checkForecasts(t, model, h)
	})
}

// FuzzAutoExog drives the auto+exog interaction: AutoARIMA(..., WithExog(X)),
// which nets X out with a provisional β before selecting d and re-estimates β per
// candidate fit. A fuzzed n×k matrix and h×k future rows stress that plumbing,
// asserting selection and the exog forecasters never panic and stay finite and
// correctly sized.
func FuzzAutoExog(f *testing.F) {
	seed := make([]float64, 48)
	for i := range seed {
		seed[i] = math.Sin(float64(i)*0.3) + float64(i)*0.02
	}
	yb := floatsToBytes(seed)
	xb := floatsToBytes(seed)
	f.Add(2, 1, 2, 1, 5, yb, xb)
	f.Add(1, 0, 1, 2, 4, yb, xb)

	f.Fuzz(func(t *testing.T, maxP, maxD, maxQ, k, h int, ydata, xdata []byte) {
		maxP, maxD, maxQ = boundOrder(maxP, 3), boundOrder(maxD, 2), boundOrder(maxQ, 3)
		k = boundOrder(k, 2) + 1  // 1..3
		h = boundOrder(h, 23) + 1 // 1..24
		series := fuzzSeries(ydata)
		if len(series) == 0 {
			return // WithExog requires at least one row
		}
		X := fuzzMatrix(xdata, len(series), k)
		futureX := fuzzMatrix(xdata, h, k)

		model, err := AutoARIMA(series, maxP, maxD, maxQ, WithExog(X))
		if err != nil {
			return
		}
		checkExogForecasts(t, model, h, futureX)
	})
}
