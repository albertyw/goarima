package goarima

// polyMul multiplies two polynomials given as coefficient slices (index = power
// of B), returning their product.
func polyMul(a, b []float64) []float64 {
	out := make([]float64, len(a)+len(b)-1)
	for i, av := range a {
		for j, bv := range b {
			out[i+j] += av * bv
		}
	}
	return out
}

// arPolynomial returns the expanded autoregressive operator
//
//	Φ_full(B) = φ(B)·(1−B)ᵈ·(1−Bᵐ)ᴰ
//
// as a coefficient slice with Φ_full[0] = 1. phi holds the AR coefficients in the
// y_t = Σ φ_j y_{t−j} convention, so φ(B) = 1 − φ₁B − … − φ_pBᵖ.
func arPolynomial(phi []float64, d, bigD, period int) []float64 {
	poly := []float64{1}
	for _, v := range phi {
		poly = polyMul(poly, []float64{1, -v}) // multiply (1 − φ_j B) term by term
	}
	for k := 0; k < d; k++ {
		poly = polyMul(poly, []float64{1, -1}) // (1 − B)
	}
	if bigD > 0 && period >= 1 {
		seasonal := make([]float64, period+1) // (1 − Bᵐ)
		seasonal[0] = 1
		seasonal[period] = -1
		for k := 0; k < bigD; k++ {
			poly = polyMul(poly, seasonal)
		}
	}
	return poly
}

// psiWeights returns the first n MA(∞) coefficients ψ₀…ψ_{n−1} of the model with
// AR coefficients phi, MA coefficients theta, and differencing (1−B)ᵈ(1−Bᵐ)ᴰ.
// They satisfy ψ(B)·Φ_full(B) = θ(B) with ψ₀ = 1, and give the h-step
// forecast-error variance σ²·Σ_{j<h} ψ_j².
func psiWeights(phi, theta []float64, d, bigD, period, n int) []float64 {
	ar := arPolynomial(phi, d, bigD, period) // ar[0] == 1
	psi := make([]float64, n)
	for j := 0; j < n; j++ {
		var v float64
		if j == 0 {
			v = 1
		} else if j <= len(theta) {
			v = theta[j-1]
		}
		// Subtract the AR feedback: ψ_j = θ_j − Σ_{i≥1} ar_i·ψ_{j−i}.
		for i := 1; i < len(ar) && i <= j; i++ {
			v -= ar[i] * psi[j-i]
		}
		psi[j] = v
	}
	return psi
}
