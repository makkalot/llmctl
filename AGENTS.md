# AGENTS.md

## What this is

`llmctl` is a single-file Go CLI (`llmctl.go`) that manages local llama.cpp model servers and exposes an OpenAI-compatible reverse proxy. Zero external Go dependencies — standard library only.

## Commands

```sh
make build          # cross-compile all 4 targets (darwin/linux × amd64/arm64) into ./bin/
make lint           # go vet + golangci-lint (default config, no .golangci.yml)
make test           # go test -v ./... (~28 tests in llmctl_test.go)
make install        # build + copy current-platform binary to /usr/local/bin/llmctl
```

Build injects version via ldflags: `-X main.appVersion=<git-tag> -X main.buildTime=<ts>`.

## Architecture

Everything lives in `llmctl.go` under `package main`. Sections are separated by comment banners:

- **Config** (~L36–185): JSON config at `~/.llmctl.json`, per-model overrides, `mergeExtraArgs`
- **Instance Registry** (~L203–240): tracks running backends in `~/.llmctl.registry.json`, includes resolved config (ctx_size, gpu_layers, extra_args, aliases)
- **Process helpers** (~L242–290): start/stop/health-check of detached llama-server processes
- **Model helpers** (~L292–530): .gguf file discovery, HuggingFace cache layout support (`:` and `/` separators), fuzzy matching, `hfRepoParts`, `findHFSnapshot`
- **Reverse Proxy** (~L555–760): OpenAI-compatible `/v1/models`, `/v1/chat/completions`, `/health`
- **CLI Commands** (~L760–1190): `load`, `unload`, `ps`, `info`, `proxy`, `pull`, `list`, `logs`, `alias`, etc.
- **Main** (~L1350–1500): CLI dispatch via switch statement

## Conventions an agent should know

- **No sub-packages.** Do not create new Go files or packages — all code goes in `llmctl.go`.
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
