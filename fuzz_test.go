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
