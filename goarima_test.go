package goarima

import (
	"fmt"
	"testing"
)

func TestARIMA(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	p, d, q := 1, 1, 1

	model, err := NewARIMA(p, d, q)
	if err != nil {
		t.Fatalf("Failed to create ARIMA model: %v", err)
	}
	if model == nil {
		t.Fatal("Failed to create ARIMA model")
	}

	err = model.Fit(data)
	if err != nil {
		t.Fatalf("Failed to fit ARIMA model: %v", err)
	}

	forecast, err := model.Forecast(5)
	if err != nil {
		t.Fatalf("Failed to forecast: %v", err)
	}
	if len(forecast) != 5 {
		t.Fatalf("Expected 5 forecast, got %d", len(forecast))
	}

	fmt.Println("Forecast:", forecast)
}
