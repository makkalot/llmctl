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

Build injects version via ldflags: `-X main.AppVersion=<git-tag> -X main.buildTime=<ts>`.

## Architecture

The public API lives at the repo root (`package main`; `main.go` with usage text + thin CLI dispatch, plus the ldflags-injected `AppVersion` var). Implementation lives in the **`internal/llmctl`** package (`package llmctl`) — module files:

- **`internal/llmctl/config.go`**: shared constants (incl. fallback `AppVersion`), `Config`/`ModelConfig`/`AutoswitchConfig`, JSON config at `~/.llmctl.json`, `LoadConfig`/save, per-model overrides, `mergeExtraArgs`, validation, key matching, `findServerBin` (unexported)
- **`internal/llmctl/registry.go`**: `Instance`/`Registry` types, persistence in `~/.llmctl.registry.json`, registry query/mutation helpers, plus process lifecycle primitives (`isRunning`, `stopProcess`, `waitForHealth`)
- **`internal/llmctl/process.go`**: log-tail and backend-exit checks
- **`internal/llmctl/models.go`**: .gguf file discovery, HuggingFace cache layout support (`:` and `/` separators), fuzzy matching, `hfRepoParts`, `findHFSnapshot`, naming helpers
- **`internal/llmctl/autoswitch.go`**: VRAM accounting (nvidia-smi), eviction planning, fallback decisions
- **`internal/llmctl/proxy.go`**: OpenAI-compatible `/v1/models`, `/v1/chat/completions`, `/health`, UI handlers, embedded `web/index.html`, event state, `StartProxy`
- **`internal/llmctl/load.go`**: `loadOptions`/`loadSpec`, mmproj resolution, `loadInstance` lifecycle
- **`internal/llmctl/commands.go`**: one exported function per CLI command (`CmdLoad`, `CmdPull`, `CmdPS`, `CmdSet`, …); the rest stays unexported
- **`internal/llmctl/helpers.go`**: small generic host/IP/flag helpers (`ExtractFlag`)
- **`internal/llmctl/web/`**: embedded UI assets

The `internal/` import restriction means implementation details stay private to this module.

## Conventions an agent should know

- **Entrypoint vs internals.** Only root `main.go` (`package main`) defines the CLI entrypoint; everything else goes in `internal/llmctl`. Do not add implementation files at the repo root. Symbols called from `main` must be exported (e.g. `llmctl.CmdLoad`, `llmctl.LoadConfig`, `llmctl.StartProxy`, `llmctl.ExtractFlag`).
- **No third-party deps.** Do not add `require` entries to `go.mod`. Use only the standard library.
- **Tests** live in `internal/llmctl/llmctl_test.go` (`package llmctl`). Covers model resolution, HF cache layout, alias/config key matching, pull target parsing, autoswitch VRAM estimation & eviction logic, and proxy model listing. Add new tests there — no external deps.
- **Runtime state** is file-based JSON in `$HOME` (`~/.llmctl.json`, `~/.llmctl.registry.json`, `~/.llmctl-logs/`).
- **Process management** detaches backends with `Setsid: true` and uses SIGTERM → SIGKILL.
- **Fuzzy matching** is used for model/instance resolution (case-insensitive substring). Don't break this contract.
- **Model names** support both `/` and `:` as separators (e.g., `Qwen3.5-27B-GGUF:UD-Q4_K_XL` and `Qwen3.5-27B-GGUF/UD-Q4_K_XL` both resolve the same way).
- **Per-model `extra_args`** are merged with global args via `MergeExtraArgs` — matching `--flag` values are replaced, new flags are appended.
- **Alias-as-name**: when loading a model via an alias, the alias becomes the instance name and API model ID. This allows loading the same model twice under different aliases/params.
- **`llmctl info <name>`** shows detailed instance info including resolved config params and aliases.
- **linux/arm64 target** is labeled "Jetson" — keep the Makefile comment if modifying build targets.
- Compiled binaries in `bin/` are gitignored.
