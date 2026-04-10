# AGENTS.md

## What this is

`llmctl` is a single-file Go CLI (`llmctl.go`, ~1400 lines) that manages local llama.cpp model servers and exposes an OpenAI-compatible reverse proxy. Zero external Go dependencies — standard library only.

## Commands

```sh
make build          # cross-compile all 4 targets (darwin/linux × amd64/arm64) into ./bin/
make lint           # go vet + golangci-lint (default config, no .golangci.yml)
make test           # go test -v ./... (no tests exist yet)
make install        # build + copy current-platform binary to /usr/local/bin/llmctl
```

Build injects version via ldflags: `-X main.appVersion=<git-tag> -X main.buildTime=<ts>`.

## Architecture

Everything lives in `llmctl.go` under `package main`. Sections are separated by comment banners:

- **Config** (~L36–131): JSON config at `~/.llmctl.json`, per-model overrides
- **Instance Registry** (~L133–217): tracks running backends in `~/.llmctl.registry.json`
- **Process helpers** (~L219–264): start/stop/health-check of detached llama-server processes
- **Model helpers** (~L266–443): .gguf file discovery, HuggingFace cache layout support, fuzzy matching
- **Reverse Proxy** (~L445–637): OpenAI-compatible `/v1/models`, `/v1/chat/completions`, `/health`
- **CLI Commands** (~L639–1197): `load`, `unload`, `ps`, `proxy`, `pull`, `list`, `logs`, `alias`, etc.
- **Main** (~L1240–1369): CLI dispatch via switch statement

## Conventions an agent should know

- **No sub-packages.** Do not create new Go files or packages — all code goes in `llmctl.go`.
- **No third-party deps.** Do not add `require` entries to `go.mod`. Use only the standard library.
- **No tests yet.** `_test.go` files don't exist. If adding tests, they go in the root alongside `llmctl.go`.
- **Runtime state** is file-based JSON in `$HOME` (`~/.llmctl.json`, `~/.llmctl.registry.json`, `~/.llmctl-logs/`).
- **Process management** detaches backends with `Setsid: true` and uses SIGTERM → SIGKILL.
- **Fuzzy matching** is used for model/instance resolution (case-insensitive substring). Don't break this contract.
- **linux/arm64 target** is labeled "Jetson" — keep the Makefile comment if modifying build targets.
- Compiled binaries in `bin/` are gitignored.
