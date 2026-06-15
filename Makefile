SHELL := /bin/bash

.PHONY:all
all: test

.PHONY:clean
clean:
	rm goarima.test c.out || true

.PHONY:install-test-deps
install-test-deps:
	curl -sSfL https://golangci-lint.run/install.sh | sh -s -- -b "$(shell go env GOPATH)/bin" v2.11.4
	go install golang.org/x/vuln/cmd/govulncheck@latest

.PHONY:test
test: install-test-deps unit
	go vet ./...
	gofmt -e -l -d -s .
	go mod tidy
	golangci-lint run ./...
	govulncheck ./...

.PHONY:unit
unit:
	go test -coverprofile=c.out -covermode=atomic ./...

.PHONY:cover
cover: test
	go tool cover -func=c.out

.PHONY:race
race:
	go test -race ./...

.PHONY:benchmark
benchmark:
	go test -bench=. -benchmem

.PHONY:example
example:
	cd example && if [ -x env/bin/python ]; then env/bin/python compare.py; else echo "no example/env; running goarima only (python3 -m venv env && env/bin/pip install -e . for the statsmodels comparison)" && go run .; fi

.PHONY:charts
charts:
	cd example && if [ -x env/bin/python ]; then env/bin/python plot_compare.py && env/bin/python plot_seasonal.py; else echo "no example/env; install matplotlib in example/env to generate charts"; fi
