.PHONY: build clean install lint test

BINARY_NAME=llmctl
VERSION=$(shell git describe --always --tags 2>/dev/null || echo "0.2.0")
BUILD_TIME=$(shell date -u '+%Y-%m-%d_%H:%M:%S')
LDFLAGS=-ldflags "-X main.appVersion=$(VERSION) -X main.buildTime=$(BUILD_TIME)"
OUTPUT_DIR=./bin

build: build-darwin-amd64 build-darwin-arm64 build-linux-amd64 build-linux-arm64

build-darwin-amd64:
	@echo "Building darwin/amd64..."
	GOOS=darwin GOARCH=amd64 go build $(LDFLAGS) -o $(OUTPUT_DIR)/$(BINARY_NAME)-darwin-amd64 .

build-darwin-arm64:
	@echo "Building darwin/arm64..."
	GOOS=darwin GOARCH=arm64 go build $(LDFLAGS) -o $(OUTPUT_DIR)/$(BINARY_NAME)-darwin-arm64 .

build-linux-amd64:
	@echo "Building linux/amd64..."
	GOOS=linux GOARCH=amd64 go build $(LDFLAGS) -o $(OUTPUT_DIR)/$(BINARY_NAME)-linux-amd64 .

build-linux-arm64:
	@echo "Building linux/arm64 (Jetson)..."
	GOOS=linux GOARCH=arm64 go build $(LDFLAGS) -o $(OUTPUT_DIR)/$(BINARY_NAME)-linux-arm64 .

clean:
	@rm -rf $(OUTPUT_DIR)

install: build
	@echo "Installing..."
	install -d /usr/local/bin
	install -m 755 $(OUTPUT_DIR)/$(BINARY_NAME)-$(shell go env GOOS)-$(shell go env GOARCH) /usr/local/bin/$(BINARY_NAME)

lint:
	go vet ./...
	golangci-lint run

test:
	go test -v ./...
