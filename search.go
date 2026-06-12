package goarima

import "math"

// searchSpace holds the inputs shared by every candidate evaluation during an
// AutoARIMA order search. d is fixed (chosen by selectD); the search ranges over
// p in 0..maxP and q in 0..maxQ.
type searchSpace struct {
	series     []float64
	d, n       int
	maxP, maxQ int
	crit       Criterion
	opts       []FitOption
}

// candidate is the cached outcome of fitting one (p,q): its criterion score and
// whether the fit succeeded.
type candidate struct {
	score float64
	ok    bool
}

// evalCandidate fits NewARIMA(p,d,q).Fit(series, opts…) and returns its criterion
// score. ok is false when (p,q) is out of bounds, is (0,0), or the fit fails
// (e.g. non-stationary/non-invertible, too few observations).
func (s searchSpace) evalCandidate(p, q int) candidate {
	if p < 0 || q < 0 || p > s.maxP || q > s.maxQ {
		return candidate{ok: false}
	}
	if p == 0 && q == 0 {
		return candidate{ok: false}
	}
	model, err := NewARIMA(p, s.d, q)
	if err != nil {
		return candidate{ok: false}
	}
	if err := model.Fit(s.series, s.opts...); err != nil {
		return candidate{ok: false}
	}
	return candidate{score: score(s.crit, s.n, model.sigma2, p, q), ok: true}
}

// gridSearch evaluates every (p,q) in 0..maxP × 0..maxQ (skipping (0,0)) and
// returns the order minimizing the criterion. It scans p-outer, q-inner and
// keeps the first strictly-smaller score, so ties break to the lowest p then the
// lowest q. bestP is -1 when no candidate could be fit.
func (s searchSpace) gridSearch() (bestP, bestQ int) {
	bestScore := math.Inf(1)
	bestP, bestQ = -1, -1
	for p := 0; p <= s.maxP; p++ {
		for q := 0; q <= s.maxQ; q++ {
			c := s.evalCandidate(p, q)
			if c.ok && c.score < bestScore {
				bestScore = c.score
				bestP, bestQ = p, q
			}
		}
	}
	return bestP, bestQ
}

// stepwiseSearch runs a Hyndman-Khandakar neighbor hill-climb: starting from a
// few seed orders it repeatedly moves to the best strictly-better neighbor until
// none improves. It typically fits far fewer candidates than the grid but is a
// heuristic and can miss the global optimum. Each (p,q) is fit at most once
// (cached). bestP is -1 when no candidate could be fit.
func (s searchSpace) stepwiseSearch() (bestP, bestQ int) {
	cache := map[[2]int]candidate{}
	evalCached := func(p, q int) candidate {
		key := [2]int{p, q}
		if c, seen := cache[key]; seen {
			return c
		}
		c := s.evalCandidate(p, q)
		cache[key] = c
		return c
	}

	bestScore := math.Inf(1)
	bestP, bestQ = -1, -1
	consider := func(p, q int) {
		c := evalCached(p, q)
		if c.ok && c.score < bestScore {
			bestScore = c.score
			bestP, bestQ = p, q
		}
	}

	// Seed orders, clamped to the maxima; (0,0) seeds are simply rejected by eval.
	for _, seed := range [][2]int{{2, 2}, {1, 0}, {0, 1}} {
		consider(min(seed[0], s.maxP), min(seed[1], s.maxQ))
	}
	if bestP < 0 {
		return -1, -1
	}

	neighbors := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}}
	for {
		curP, curQ := bestP, bestQ
		for _, n := range neighbors {
			consider(curP+n[0], curQ+n[1])
		}
		if bestP == curP && bestQ == curQ {
			break // no neighbor improved on the current best
		}
	}
	return bestP, bestQ
}
