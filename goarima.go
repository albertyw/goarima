package goarima

// TimeSeries represents a time series of data for ARIMA to operate on
type TimeSeries struct {
	Data []float64
}

// ARIMA is the configuration of a model which operates on a TimeSeries
type ARIMA struct {
	// The order of the AR model
	p int
	// The order of the I model
	d int
	// The order of the MA model
	q int
}

// NewARIMA creates a new ARIMA model with the given parameters
func NewARIMA(p, d, q int) ARIMA {
	return ARIMA{
		p: p,
		d: d,
		q: q,
	}
}
