# AGENTS.md

## What this is

`llmctl` is a single-file Go CLI (`llmctl.go`) that manages local llama.cpp model servers (each run as a Docker container) and exposes an OpenAI-compatible reverse proxy. Zero external Go dependencies — standard library only.

## Commands

```sh
make build          # cross-compile all 4 targets (darwin/linux × amd64/arm64) into ./bin/
make lint           # go vet + golangci-lint (default config, no .golangci.yml)
make test           # go test -v ./... (~64 tests in llmctl_test.go)
make install        # build + copy current-platform binary to /usr/local/bin/llmctl
make systemd-install    # install + enable systemd system service (/etc/systemd/system/llmctl.service, runs as $SUDO_USER)
make systemd-uninstall  # disable + remove systemd system service
```

Build injects version via ldflags: `-X main.appVersion=<git-tag> -X main.buildTime=<ts>`.

## Architecture

Everything lives in `llmctl.go` under `package main`. Sections are separated by comment banners:

- **Config** (~L36–185): JSON config at `~/.llmctl.json`, per-model overrides, `mergeExtraArgs`
- **Instance Registry** (~L203–240): tracks running backends in `~/.llmctl.registry.json`, includes resolved config (ctx_size, gpu_layers, extra_args, aliases)
- **Process helpers**: proxy PID helpers (`isRunning`, `stopProcess`)
- **Docker helpers**: `docker` CLI wrapper (`dockerExec`, stubbable in tests), live-container listing, `dockerRunArgs`, `backendMounts`, stop/remove with log capture
- **Custom llama.cpp builds**: `llmctl build` / `builds` — `builds` config, CLI shortcuts (`build <url>` auto-adds via `parseBuildSource`, `build add/rm`, `load --build/--network`), ref resolution (`git ls-remote`, `#PR`), host-side source fetch/copy, `dockerBuildArgs`, retention (`imagesToPrune`). `git` is called via `gitExec` (stubbable). The Dockerfile is `docker/llama-server.Dockerfile`, embedded with `go:embed` — keep its `ARG`s in sync with `dockerBuildArgs` (a test checks this).
- **Model helpers** (~L292–530): .gguf file discovery, HuggingFace cache layout support (`:` and `/` separators), fuzzy matching, `hfRepoParts`, `findHFSnapshot`
- **Reverse Proxy** (~L555–760): OpenAI-compatible `/v1/models`, `/v1/chat/completions`, `/health`
- **CLI Commands** (~L760–1190): `load`, `unload`, `ps`, `info`, `proxy`, `pull`, `list`, `logs`, `alias`, etc.
- **Main** (~L1350–1500): CLI dispatch via switch statement

## Conventions an agent should know

- **No sub-packages.** Do not create new Go files or packages — all code goes in `llmctl.go`. Non-Go assets embedded with `go:embed` (`web/index.html`, `docker/llama-server.Dockerfile`) are fine.
- **No third-party deps.** Do not add `require` entries to `go.mod`. Use only the standard library.
- **Tests** live in `llmctl_test.go` in the root (~64 tests). Covers model resolution, HF cache layout, docker run args / mounts / registry cleanup (docker stubbed via `dockerExec`), alias/config key matching, pull target parsing, autoswitch VRAM estimation & eviction logic, and proxy model listing. Add new tests there — same `package main`, no external deps.
- **Runtime state** is file-based JSON in `$HOME` (`~/.llmctl.json`, `~/.llmctl.registry.json`, `~/.llmctl-logs/`).
- **Backends are Docker containers** named `llmctl-<instance>`, labelled `llmctl.instance=<instance>`, started without `--rm` (llmctl saves logs to `~/.llmctl-logs/` and removes them). Host paths are bind-mounted at the same path so model paths are identical inside the container. Containers are hardened (`--cap-drop ALL`, `no-new-privileges`, read-only rootfs, pids limit). Network modes (`docker.network`): `none` (default; llama-server on a unix socket in `~/.llmctl-run/<container>/`, proxy uses a unix-dialing transport), `internal` (`llmctl-internal` docker network, proxy dials container IP), `bridge` (port published on `127.0.0.1`). Always reach backends via `instanceBackend(inst)` — never build `127.0.0.1:<port>` URLs directly. In `none`/`internal`, `-hf` loads are prefetched on the host (`prefetchHF`). Talk to Docker only via the CLI (`dockerExec`) — no Docker SDK.
- **`CleanDead` must not wipe the registry when `docker ps` fails** (daemon unreachable) — it returns early.
- **Fuzzy matching** is used for model/instance resolution (case-insensitive substring). Don't break this contract.
- **Model names** support both `/` and `:` as separators (e.g., `Qwen3.5-27B-GGUF:UD-Q4_K_XL` and `Qwen3.5-27B-GGUF/UD-Q4_K_XL` both resolve the same way).
- **Per-model `extra_args`** are merged with global args via `mergeExtraArgs` — matching `--flag` values are replaced, new flags are appended.
- **Alias-as-name**: when loading a model via an alias, the alias becomes the instance name and API model ID. This allows loading the same model twice under different aliases/params.
- **`llmctl info <name>`** shows detailed instance info including resolved config params and aliases.
- **linux/arm64 target** is labeled "Jetson" — keep the Makefile comment if modifying build targets.
- Compiled binaries in `bin/` are gitignored.
