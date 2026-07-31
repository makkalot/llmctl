.PHONY: build clean install lint test systemd-install systemd-uninstall

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

SERVICE_FILE=llmctl.service
SERVICE_DEST=/etc/systemd/system/llmctl.service
SERVICE_USER=$(or $(SUDO_USER),$(USER))

systemd-install: install
	@echo "Installing systemd system service (user: $(SERVICE_USER))..."
	sed 's/@SERVICE_USER@/$(SERVICE_USER)/' $(SERVICE_FILE) > $(SERVICE_DEST)
	systemctl daemon-reload
	systemctl enable --now llmctl.service
	@echo "Service installed and started."
	@echo "  sudo systemctl status llmctl   # check status"
	@echo "  journalctl -u llmctl -f        # follow logs"
	@echo "  journalctl -u llmctl -n 100    # last 100 lines"

systemd-uninstall:
	@echo "Removing systemd system service..."
	-systemctl disable --now llmctl.service
	rm -f $(SERVICE_DEST)
	systemctl daemon-reload
	@echo "Service removed."
