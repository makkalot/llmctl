# AGENTS.md

## What this is

`llmctl` is a Go CLI that manages local llama.cpp model servers and exposes an OpenAI-compatible reverse proxy. Zero external Go dependencies — standard library only.

## Commands

```sh
make build          # cross-compile all 4 targets (darwin/linux × amd64/arm64) into ./bin/
make lint           # go vet + golangci-lint (default config, no .golangci.yml)
make test           # go test -v ./... (~28 tests in llmctl_test.go)
make install        # build + copy current-platform binary to /usr/local/bin/llmctl
make systemd-install    # install + enable systemd system service (/etc/systemd/system/llmctl.service, runs as $SUDO_USER)
make systemd-uninstall  # disable + remove systemd system service
```

Build injects version via ldflags: `-X main.appVersion=<git-tag> -X main.buildTime=<ts>`.

## Architecture

Code is split into same-package module files, all under `package main` (no sub-packages). Use this grouping when adding code:

- **`config.go`**: shared constants, `Config`/`ModelConfig`/`AutoswitchConfig`, JSON config at `~/.llmctl.json`, load/save, per-model overrides, `mergeExtraArgs`, validation, key matching, `findServerBin`
- **`registry.go`**: `Instance`/`Registry` types, persistence in `~/.llmctl.registry.json`, registry query/mutation helpers, plus process lifecycle primitives (`isRunning`, `stopProcess`, `waitForHealth`)
- **`process.go`**: log-tail and backend-exit checks
- **`models.go`**: .gguf file discovery, HuggingFace cache layout support (`:` and `/` separators), fuzzy matching, `hfRepoParts`, `findHFSnapshot`, naming helpers
- **`autoswitch.go`**: VRAM accounting (nvidia-smi), eviction planning, fallback decisions
- **`proxy.go`**: OpenAI-compatible `/v1/models`, `/v1/chat/completions`, `/health`, UI handlers, embedded `web/index.html`, event state, `startProxy`
- **`load.go`**: `loadOptions`/`loadSpec`, mmproj resolution, `loadInstance` lifecycle
- **`commands.go`**: one function per CLI command (`cmdLoad`, `cmdPull`, `cmdPS`, `cmdSet`, …)
- **`helpers.go`**: small generic host/IP/flag helpers
- **`main.go`**: usage text and thin CLI dispatch switch

`llmctl.go` keeps only the package declaration and this module map.

## Conventions an agent should know

- **No sub-packages.** All code lives in same-`package main` files using the module grouping above — do not create sub-packages or add new directories.
- **No third-party deps.** Do not add `require` entries to `go.mod`. Use only the standard library.
- **Tests** live in `llmctl_test.go` in the root (~28 tests). Covers model resolution, HF cache layout, alias/config key matching, pull target parsing, autoswitch VRAM estimation & eviction logic, and proxy model listing. Add new tests there — same `package main`, no external deps.
- **Runtime state** is file-based JSON in `$HOME` (`~/.llmctl.json`, `~/.llmctl.registry.json`, `~/.llmctl-logs/`).
- **Process management** detaches backends with `Setsid: true` and uses SIGTERM → SIGKILL.
- **Fuzzy matching** is used for model/instance resolution (case-insensitive substring). Don't break this contract.
- **Model names** support both `/` and `:` as separators (e.g., `Qwen3.5-27B-GGUF:UD-Q4_K_XL` and `Qwen3.5-27B-GGUF/UD-Q4_K_XL` both resolve the same way).
- **Per-model `extra_args`** are merged with global args via `mergeExtraArgs` — matching `--flag` values are replaced, new flags are appended.
- **Alias-as-name**: when loading a model via an alias, the alias becomes the instance name and API model ID. This allows loading the same model twice under different aliases/params.
- **`llmctl info <name>`** shows detailed instance info including resolved config params and aliases.
- **linux/arm64 target** is labeled "Jetson" — keep the Makefile comment if modifying build targets.
- Compiled binaries in `bin/` are gitignored.
