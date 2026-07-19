package goarima

import (
	"math"
	"math/rand"
	"testing"
)

func TestValidateExogMatrix(t *testing.T) {
	good := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	k, err := validateExogMatrix(good, 3)
	if err != nil || k != 2 {
		t.Fatalf("good matrix: k=%d err=%v", k, err)
	}
	if _, err := validateExogMatrix(good, 4); err == nil {
		t.Error("row-count mismatch should error")
	}
	ragged := [][]float64{{1, 2}, {3}}
	if _, err := validateExogMatrix(ragged, 2); err == nil {
		t.Error("ragged rows should error")
	}
	nan := [][]float64{{math.NaN()}}
	if _, err := validateExogMatrix(nan, 1); err == nil {
		t.Error("non-finite entry should error")
	}
}

func TestOLSBetaRecoversKnownSlope(t *testing.T) {
	// y = 3*x1 - 2*x2 exactly; OLS must recover (3, -2).
	dX := [][]float64{{1, 0}, {0, 1}, {1, 1}, {2, 1}, {1, 3}}
	dy := make([]float64, len(dX))
	for i, r := range dX {
		dy[i] = 3*r[0] - 2*r[1]
	}
	beta, err := olsBeta(dy, dX)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(beta[0]-3) > 1e-9 || math.Abs(beta[1]+2) > 1e-9 {
		t.Fatalf("got beta=%v, want [3 -2]", beta)
	}
}

func TestDifferenceExogMatchesColumnwise(t *testing.T) {
	X := [][]float64{{1, 10}, {2, 14}, {4, 20}, {7, 28}}
	got := differenceExog(X, 1, 0, 0) // first regular difference
	want := [][]float64{{1, 4}, {2, 6}, {3, 8}}
	if len(got) != len(want) {
		t.Fatalf("rows: got %d want %d", len(got), len(want))
	}
	for i := range want {
		for j := range want[i] {
			if math.Abs(got[i][j]-want[i][j]) > 1e-12 {
				t.Fatalf("row %d col %d: got %v want %v", i, j, got[i][j], want[i][j])
			}
		}
	}
}

func TestFitWithExogRecoversBeta(t *testing.T) {
	// y_t = 2*x_t + AR(1) errors with phi=0.5 and white-noise innovations.
	rng := rand.New(rand.NewSource(20260622))
	n := 400
	x := make([]float64, n)
	y := make([]float64, n)
	var eta float64
	for i := 0; i < n; i++ {
		x[i] = math.Sin(float64(i) / 5.0)
		eta = 0.5*eta + rng.NormFloat64()*0.3
		y[i] = 2*x[i] + eta
	}
	X := make([][]float64, n)
	for i := range X {
		X[i] = []float64{x[i]}
	}
	m, err := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Fit(y, WithExog(X)); err != nil {
		t.Fatal(err)
	}
	b := m.Beta()
	if len(b) != 1 || math.Abs(b[0]-2) > 0.2 {
		t.Fatalf("got beta=%v, want ~[2]", b)
	}
	if len(m.Phi()) != 1 || math.Abs(m.Phi()[0]-0.5) > 0.2 {
		t.Fatalf("got phi=%v, want ~[0.5]", m.Phi())
	}
}

func TestFitWithoutExogLeavesBetaEmpty(t *testing.T) {
	m, _ := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if err := m.Fit([]float64{1, 2, 1, 2, 1, 2, 1, 2, 1, 2}); err != nil {
		t.Fatal(err)
	}
	if len(m.Beta()) != 0 {
		t.Fatalf("Beta should be empty without exog, got %v", m.Beta())
	}
}

func TestForecastExogRespondsToFutureX(t *testing.T) {
	n := 120
	x := make([]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = float64(i % 7)
		y[i] = 5*x[i] + math.Sin(float64(i))
	}
	X := make([][]float64, n)
	for i := range X {
		X[i] = []float64{x[i]}
	}
	m, _ := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if err := m.Fit(y, WithExog(X)); err != nil {
		t.Fatal(err)
	}
	low := [][]float64{{0}, {0}, {0}}
	high := [][]float64{{10}, {10}, {10}}
	fLow, err := m.Forecast(3, WithFutureExog(low))
	if err != nil {
		t.Fatal(err)
	}
	fHigh, err := m.Forecast(3, WithFutureExog(high))
	if err != nil {
		t.Fatal(err)
	}
	// Each unit of x adds ~beta (~5) to the forecast; 10 units >> 0 units.
	for i := 0; i < 3; i++ {
		if fHigh[i] <= fLow[i]+30 {
			t.Fatalf("step %d: high=%v not sufficiently above low=%v", i, fHigh[i], fLow[i])
		}
	}
}

func TestForecastExogValidatesShape(t *testing.T) {
	n := 60
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := range X {
		X[i] = []float64{float64(i), float64(i % 3)}
		y[i] = float64(i) + math.Sin(float64(i))
	}
	m, _ := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if err := m.Fit(y, WithExog(X)); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Forecast(3, WithFutureExog([][]float64{{1, 1}, {1, 1}})); err == nil {
		t.Error("wrong row count should error")
	}
	if _, err := m.Forecast(2, WithFutureExog([][]float64{{1}, {1}})); err == nil {
		t.Error("wrong column count should error")
	}
}

func TestOLSBetaErrors(t *testing.T) {
	if _, err := olsBeta(nil, nil); err == nil {
		t.Error("empty design matrix should error")
	}
	// Two columns but only two rows: rows <= cols.
	if _, err := olsBeta([]float64{1, 2}, [][]float64{{1, 0}, {0, 1}}); err == nil {
		t.Error("too few rows for the regressors should error")
	}
}

func TestValidateExogMatrixEmptyShapes(t *testing.T) {
	if _, err := validateExogMatrix(nil, 0); err == nil {
		t.Error("zero rows should error")
	}
	if _, err := validateExogMatrix([][]float64{{}}, 1); err == nil {
		t.Error("zero-width rows should error")
	}
}

func TestFitWithExogTooShortErrors(t *testing.T) {
	// More regressors than usable differenced rows: the OLS step must fail and Fit
	// surface it.
	y := []float64{1, 2, 3, 4, 5, 6}
	X := make([][]float64, len(y))
	for i := range X {
		X[i] = []float64{float64(i), float64(i * i), float64(i * i * i), float64(i + 1), float64(2 * i)}
	}
	m, _ := NewARIMA(Order{P: 0, D: 1, Q: 0})
	if err := m.Fit(y, WithExog(X)); err == nil {
		t.Error("exog regression with too few rows should error")
	}
}

func TestForecastExogInvalidHorizon(t *testing.T) {
	n := 40
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := range X {
		X[i] = []float64{float64(i % 3)}
		y[i] = float64(i%3) + math.Sin(float64(i))
	}
	m, _ := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if err := m.Fit(y, WithExog(X)); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Forecast(0, WithFutureExog([][]float64{{0}})); err == nil {
		t.Error("non-positive horizon should error")
	}
	if _, err := m.ForecastInterval(0, 0.95, WithFutureExog([][]float64{{0}})); err == nil {
		t.Error("non-positive horizon should error")
	}
}

func TestExogCSSRefinement(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	n := 220
	x := make([]float64, n)
	y := make([]float64, n)
	var eta float64
	for i := 0; i < n; i++ {
		x[i] = math.Sin(float64(i) / 3.0)
		eta = 0.5*eta + rng.NormFloat64()*0.3
		y[i] = 2.5*x[i] + eta
	}
	X := make([][]float64, n)
	for i := range X {
		X[i] = []float64{x[i]}
	}
	m, _ := NewARIMA(Order{P: 1, D: 0, Q: 1})
	if err := m.Fit(y, WithExog(X), WithMethod(CSS)); err != nil {
		t.Fatal(err)
	}
	if b := m.Beta(); len(b) != 1 || math.Abs(b[0]-2.5) > 0.3 {
		t.Fatalf("CSS exog beta=%v, want ~[2.5]", b)
	}
	f, err := m.Forecast(3, WithFutureExog([][]float64{{0.1}, {0.2}, {0.3}}))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range f {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite CSS exog forecast %v", f)
		}
	}
}

func TestSeasonalFitWithExogIsFinite(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	m := 12
	n := 180
	x := make([]float64, n)
	y := make([]float64, n)
	var eta float64
	for i := 0; i < n; i++ {
		x[i] = float64(i%m) / float64(m)
		season := math.Sin(2 * math.Pi * float64(i) / float64(m))
		eta = 0.4*eta + rng.NormFloat64()*0.3
		y[i] = 3*x[i] + season + eta
	}
	X := make([][]float64, n)
	for i := range X {
		X[i] = []float64{x[i]}
	}
	model, err := NewSARIMA(Order{P: 1, D: 0, Q: 0}, SeasonalOrder{P: 1, D: 0, Q: 0, Period: m})
	if err != nil {
		t.Fatal(err)
	}
	if err := model.Fit(y, WithExog(X)); err != nil {
		t.Fatal(err)
	}
	if b := model.Beta(); len(b) != 1 || math.Abs(b[0]-3) > 0.6 {
		t.Fatalf("seasonal exog beta=%v, want ~[3]", b)
	}
	futureX := make([][]float64, m)
	for i := range futureX {
		futureX[i] = []float64{float64(i%m) / float64(m)}
	}
	f, err := model.Forecast(m, WithFutureExog(futureX))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range f {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite seasonal exog forecast %v", f)
		}
	}

	// The joint refinement must also handle seasonal factors + β together.
	mleModel, _ := NewSARIMA(Order{P: 1, D: 0, Q: 0}, SeasonalOrder{P: 1, D: 0, Q: 0, Period: m})
	if err := mleModel.Fit(y, WithExog(X), WithMethod(MLE)); err != nil {
		t.Fatal(err)
	}
	fm, err := mleModel.Forecast(m, WithFutureExog(futureX))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range fm {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite seasonal exog MLE forecast %v", fm)
		}
	}
}

func TestSeasonalExogDifferencing(t *testing.T) {
	// With D=1,m=4,d=1, differenceExog must drop m+1 rows (seasonal then regular).
	X := make([][]float64, 20)
	for i := range X {
		X[i] = []float64{float64(i), float64(i * i)}
	}
	got := differenceExog(X, 1, 1, 4)
	if want := 20 - 4 - 1; len(got) != want {
		t.Fatalf("rows: got %d want %d", len(got), want)
	}
}

func TestExogMLEImprovesOrMatches(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	n := 300
	x := make([]float64, n)
	y := make([]float64, n)
	var eta float64
	for i := 0; i < n; i++ {
		x[i] = math.Cos(float64(i) / 4.0)
		eta = 0.6*eta + rng.NormFloat64()*0.4
		y[i] = -1.5*x[i] + eta
	}
	X := make([][]float64, n)
	for i := range X {
		X[i] = []float64{x[i]}
	}
	hr, _ := NewARIMA(Order{P: 1, D: 0, Q: 1})
	if err := hr.Fit(y, WithExog(X)); err != nil {
		t.Fatal(err)
	}
	mle, _ := NewARIMA(Order{P: 1, D: 0, Q: 1})
	if err := mle.Fit(y, WithExog(X), WithMethod(MLE)); err != nil {
		t.Fatal(err)
	}
	// β stays sensible and finite under joint refinement.
	if b := mle.Beta(); len(b) != 1 || math.Abs(b[0]+1.5) > 0.3 {
		t.Fatalf("MLE beta=%v, want ~[-1.5]", b)
	}
	// Refined residual variance is no worse than the HR seed's.
	if mle.Sigma2() > hr.Sigma2()+1e-9 {
		t.Fatalf("MLE sigma2 %v worse than HR %v", mle.Sigma2(), hr.Sigma2())
	}
	f, err := mle.Forecast(4, WithFutureExog([][]float64{{0.1}, {0.2}, {0.3}, {0.4}}))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range f {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite forecast %v", f)
		}
	}
}

func TestAutoARIMAWithExogSelectsAndForecasts(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	n := 240
	x := make([]float64, n)
	y := make([]float64, n)
	var eta float64
	for i := 0; i < n; i++ {
		x[i] = math.Sin(float64(i) / 6.0)
		eta = 0.5*eta + rng.NormFloat64()*0.3
		y[i] = 4*x[i] + eta
	}
	X := make([][]float64, n)
	for i := range X {
		X[i] = []float64{x[i]}
	}
	m, err := AutoARIMA(y, Bounds{MaxP: 3, MaxD: 1, MaxQ: 3}, WithExog(X))
	if err != nil {
		t.Fatal(err)
	}
	if b := m.Beta(); len(b) != 1 || math.Abs(b[0]-4) > 0.5 {
		t.Fatalf("auto exog beta=%v, want ~[4]", b)
	}
	f, err := m.Forecast(3, WithFutureExog([][]float64{{0.1}, {0.2}, {0.3}}))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range f {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite auto-exog forecast %v", f)
		}
	}
}

func TestAutoSARIMAWithExog(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	m := 12
	n := 264
	// The covariate must not be period-m, or seasonal differencing (1−Bᵐ)
	// annihilates it; use a slow non-seasonal cycle (period ~107).
	x := make([]float64, n)
	y := make([]float64, n)
	var eta float64
	for i := 0; i < n; i++ {
		x[i] = math.Sin(float64(i) / 17.0)
		season := math.Sin(2 * math.Pi * float64(i) / float64(m))
		eta = 0.4*eta + rng.NormFloat64()*0.3
		y[i] = 3*x[i] + season + eta
	}
	X := make([][]float64, n)
	for i := range X {
		X[i] = []float64{x[i]}
	}
	model, err := AutoSARIMA(y, Bounds{MaxP: 2, MaxD: 1, MaxQ: 2}, SeasonalBounds{MaxP: 1, MaxQ: 1, Period: m}, WithExog(X))
	if err != nil {
		t.Fatal(err)
	}
	if len(model.Beta()) != 1 {
		t.Fatalf("expected one beta, got %v", model.Beta())
	}
	futureX := make([][]float64, m)
	for i := range futureX {
		futureX[i] = []float64{math.Sin(float64(n+i) / 17.0)}
	}
	f, err := model.Forecast(m, WithFutureExog(futureX))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range f {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("non-finite auto-seasonal-exog forecast %v", f)
		}
	}
}

func TestAutoExogInvalidMatrixErrors(t *testing.T) {
	y := make([]float64, 40)
	for i := range y {
		y[i] = math.Sin(float64(i))
	}
	badX := [][]float64{{1}, {2}} // wrong row count
	if _, err := AutoARIMA(y, Bounds{MaxP: 2, MaxD: 1, MaxQ: 2}, WithExog(badX)); err == nil {
		t.Error("AutoARIMA with mismatched exog rows should error")
	}
	if _, err := AutoSARIMA(y, Bounds{MaxP: 2, MaxD: 1, MaxQ: 2}, SeasonalBounds{MaxP: 1, MaxQ: 1, Period: 12}, WithExog(badX)); err == nil {
		t.Error("AutoSARIMA with mismatched exog rows should error")
	}
}

func TestForecastMethodMismatchErrors(t *testing.T) {
	n := 60
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := range X {
		X[i] = []float64{float64(i % 4)}
		y[i] = 2*float64(i%4) + math.Sin(float64(i))
	}
	exog, _ := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if err := exog.Fit(y, WithExog(X)); err != nil {
		t.Fatal(err)
	}
	if _, err := exog.Forecast(3); err == nil {
		t.Error("Forecast on an exog model should error")
	}
	if _, err := exog.ForecastInterval(3, 0.95); err == nil {
		t.Error("ForecastInterval on an exog model should error")
	}
	plain, _ := NewARIMA(Order{P: 1, D: 0, Q: 0})
	if err := plain.Fit(y); err != nil {
		t.Fatal(err)
	}
	if _, err := plain.Forecast(3, WithFutureExog([][]float64{{1}, {1}, {1}})); err == nil {
		t.Error("Forecast with WithFutureExog on a non-exog model should error")
	}
	if _, err := plain.ForecastInterval(3, 0.95, WithFutureExog([][]float64{{1}, {1}, {1}})); err == nil {
		t.Error("ForecastInterval with WithFutureExog on a non-exog model should error")
	}
}
